"""Read the three final historical ranking files without API or model loading."""
import hashlib
import json
from pathlib import Path


HISTORY = Path(__file__).resolve().parent / "history"
STAGES = {
    "initial": "initial_prompts_ranked_no_agg.json",
    "round1": "fused_prompts_ranked_no_agg.json",
    "round2": "fused_prompts_round2_ranked_no_agg.json",
}


def load_history(directory=HISTORY):
    stages = {}
    for stage, filename in STAGES.items():
        rows = json.loads((Path(directory) / filename).read_text())
        for row in rows:
            digest = hashlib.md5(row["prompt_text"].encode()).hexdigest()[:8]
            if digest != row["prompt_hash"]:
                raise ValueError(f"Prompt hash mismatch in {filename}: {row['prompt_hash']}")
        stages[stage] = rows
    return stages


def summary(directory=HISTORY):
    stages = load_history(directory)
    return {stage: {"count": len(rows), "best_score": max(row["score"] for row in rows),
                    "best_prompt_hash": max(rows, key=lambda row: row["score"])["prompt_hash"]}
            for stage, rows in stages.items()}
