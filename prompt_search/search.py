"""Inspect archived results or explicitly start a fresh 30 -> 15 -> 8 search."""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

from prompt_search.history import HISTORY, STAGES, summary


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SearchConfig:
    initial_count: int = 30
    round1_target: int = 15
    round2_target: int = 8
    round1_max_combinations: int = 60
    round2_max_combinations: int = 70
    initial_temperature: float = 1.5
    round1_temperature: float = 1.0
    round2_temperature: float = 0.5


def write_rankings(directory, stage, records):
    import pandas as pd
    rows = sorted(records, key=lambda row: row["score"], reverse=True)
    for index, row in enumerate(rows, start=1):
        # Preserve evaluation/generation IDs; rank is a separate field.
        row["rank"] = index
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / STAGES[stage]).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(directory / Path(STAGES[stage]).with_suffix(".csv"), index=False)
    return rows


def rank_combinations(records, embeddings, group_size):
    import numpy as np
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    distance = 1 - normalized @ normalized.T
    np.fill_diagonal(distance, 0)
    ranked = []
    for combo in combinations(range(len(records)), group_size):
        distance_sum = sum(float(distance[i, j]) for i, j in combinations(combo, 2))
        ranked.append({"indices": combo, "source_prompt_ids": [records[i]["prompt_id"] for i in combo],
                       "source_prompt_hashes": [records[i]["prompt_hash"] for i in combo],
                       "distance_sum": distance_sum})
    ranked.sort(key=lambda item: item["distance_sum"], reverse=True)
    return distance, ranked


def generate_initial(client, template, directory, config=None):
    """Generate once per requested slot; archive duplicates without refilling."""
    from prompt_search.client import chat
    config = config or SearchConfig()
    seen, candidates, attempts = set(), [], []
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(config.initial_count):
        text = chat(client, template.generate_prompt_template(), config.initial_temperature)
        digest = hashlib.md5(text.encode()).hexdigest()[:8]
        duplicate = digest in seen
        attempts.append({"attempt": index + 1, "prompt_hash": digest, "duplicate": duplicate})
        if not duplicate:
            candidates.append({"prompt_id": index + 1, "prompt_hash": digest, "prompt_text": text})
            seen.add(digest)
        (directory / "initial_candidates.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n")
        (directory / "initial_generation_attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
    return candidates, seen


def evaluate_initial(candidates, evaluator, directory):
    records = []
    for candidate in candidates:
        result = evaluator.evaluate(candidate["prompt_text"])
        records.append({**result, "prompt_id": candidate["prompt_id"]})
        write_rankings(directory, "initial", records)
    write_rankings(directory, "initial", records)
    return records


def fuse_round(records, stage, group_size, target, max_combinations, temperature,
               directory, evaluator, embedder, client, template, seen):
    import pandas as pd
    from prompt_search.client import chat
    directory = Path(directory)
    valid = [row for row in records if math.isfinite(row["score"]) and row["score"] >= 0]
    if len(valid) < group_size:
        write_rankings(directory, stage, [])
        return [], {"status": "insufficient_valid_parents", "target": target, "actual": 0}
    # Original notebook used batches of four in round 1 and three in round 2.
    embeddings = embedder.mean([row["prompt_text"] for row in valid], batch_size=4 if stage == "round1" else 3)
    distance, ranked = rank_combinations(valid, embeddings, group_size)
    pd.DataFrame(distance, index=[r["prompt_id"] for r in valid],
                 columns=[r["prompt_id"] for r in valid]).to_csv(directory / f"{stage}_distances.csv")
    (directory / f"{stage}_combinations.json").write_text(json.dumps(ranked, indent=2) + "\n")
    baseline = max(row["score"] for row in valid)
    kept, attempts = [], []
    for rank, combo in enumerate(ranked[:max_combinations], start=1):
        parents = [valid[i] for i in combo["indices"]]
        fused = chat(client, template.aggregation_prompt_template(
            [row["prompt_text"] for row in parents], [row["score"] for row in parents]), temperature)
        digest = hashlib.md5(fused.encode()).hexdigest()[:8]
        if digest in seen:
            attempts.append({**combo, "combo_rank": rank, "prompt_hash": digest, "status": "duplicate"})
            (directory / f"{stage}_attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
            continue
        seen.add(digest)
        result = evaluator.evaluate(fused)
        accepted = math.isfinite(result["score"]) and result["score"] > baseline
        attempt = {**combo, "combo_rank": rank, "prompt_hash": digest, "score": result["score"],
                   "baseline": baseline, "accepted": accepted}
        attempts.append(attempt)
        (directory / f"{stage}_attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
        if accepted:
            kept.append({**result, "prompt_id": len(kept) + 1, "source_prompt_ids": combo["source_prompt_ids"],
                         "source_prompt_hashes": combo["source_prompt_hashes"],
                         "combo_distance_sum": combo["distance_sum"], "combo_rank": rank})
            write_rankings(directory, stage, kept)
            if len(kept) == target:
                break
    write_rankings(directory, stage, kept)
    return kept, {"target": target, "actual": len(kept), "baseline": baseline,
                  "attempts": len(attempts), "status": "complete" if len(kept) == target else "partial"}


def run_search(output_root, evaluation_config, config=None):
    import random
    import numpy as np
    from data_processing.embedding import BiomedicalEmbedder
    from prompt_search.client import make_client
    from prompt_search.evaluation import PromptEvaluator
    from prompt_search.templates import ManualMicrobiomePrompter
    config = config or SearchConfig()
    random.seed(42)
    np.random.seed(42)
    client = make_client()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = Path(output_root).resolve() / timestamp
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(json.dumps({"search": asdict(config), "evaluation": asdict(evaluation_config)}, indent=2) + "\n")
    evaluator = PromptEvaluator(directory / "evaluations", evaluation_config, client=client)
    template = ManualMicrobiomePrompter()
    candidates, seen = generate_initial(client, template, directory, config)
    initial = evaluate_initial(candidates, evaluator, directory)
    embedder = BiomedicalEmbedder(evaluation_config.embed_model, evaluation_config.device)
    stages = {"initial": initial}
    status = {"initial": {"target": config.initial_count, "actual": len(initial)}}
    for stage, group_size, target, limit, temperature, parent_stage in [
        ("round1", 4, config.round1_target, config.round1_max_combinations, config.round1_temperature, "initial"),
        ("round2", 3, config.round2_target, config.round2_max_combinations, config.round2_temperature, "round1"),
    ]:
        stages[stage], status[stage] = fuse_round(stages[parent_stage], stage, group_size, target, limit,
                                                temperature, directory, evaluator, embedder, client, template, seen)
    all_results = [row for rows in stages.values() for row in rows if row["score"] >= 0]
    if all_results:
        best = max(all_results, key=lambda row: row["score"])
        (directory / "best_prompt.txt").write_text(best["prompt_text"] + "\n")
        status["best_prompt_hash"] = best["prompt_hash"]
        status["best_score"] = best["score"]
    (directory / "summary.json").write_text(json.dumps(status, indent=2) + "\n")
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    history_parser = commands.add_parser("history", help="Offline summary of archived prompt rankings")
    history_parser.add_argument("--directory", type=Path, default=HISTORY)
    online = commands.add_parser("run", help="Start online DeepSeek generation and GCN scoring")
    online.add_argument("--output-dir", type=Path, default=ROOT / "outputs/prompt_search")
    online.add_argument("--validation-csv", type=Path, default=ROOT / "data_processing/data/prompt_validation.csv")
    online.add_argument("--embed-model", default=None)
    online.add_argument("--device", default="cuda")
    online.add_argument("--epochs", type=int, default=2000)
    online.add_argument("--patience", type=int, default=100)
    args = parser.parse_args(argv)
    if args.command == "history":
        print(json.dumps(summary(args.directory), indent=2))
    else:
        from data_processing.embedding import DEFAULT_EMBED_MODEL
        from prompt_search.evaluation import EvaluationConfig
        profile = EvaluationConfig(validation_csv=str(args.validation_csv),
                                   embed_model=args.embed_model or DEFAULT_EMBED_MODEL,
                                   device=args.device, epochs=args.epochs, patience=args.patience)
        print(run_search(args.output_dir, profile))


if __name__ == "__main__":
    main()
