"""Check manuscript claims, archived metrics, cohort inputs and new explicit presets."""
import ast
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from model.data import load_dataset
from model.engine import TrainingConfig, run_experiment
from model.networks import VARIANTS, MetaDual
from model.splits import split_fixed_test_stratified


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    work = ROOT / "verification/work" / stamp
    work.mkdir(parents=True)
    report = {"status": "running", "external_api_calls": 0, "checks": {}}
    manuscript = json.loads((ROOT / "docs/manuscript_reference.json").read_text())
    cohorts = {}
    for cohort, expected_counts, dimensions in [
        ("primary", [152, 298, 108, 2636, 342, 238, 238], 903),
        ("independent", [312, 560, 433, 2613, 762, 283, 206, 856], 2557),
    ]:
        profile = json.loads((ROOT / "configs" / f"{cohort}.json").read_text())
        assert set(profile["training"]) == {field.name for field in fields(TrainingConfig)}
        config = TrainingConfig(**profile["training"])
        assert config.metric == "braycurtis" and config.K == 3 and config.repeats == 5
        data = load_dataset(ROOT / profile["data"])
        claim = manuscript[cohort]
        assert data["X_otu_abundance"].shape == (claim["samples"], dimensions)
        assert data["X_text_embedding"].shape == (claim["samples"], 1536)
        assert data["y"].shape == (claim["samples"], claim["classes"])
        assert data["y"].sum(0).tolist() == expected_counts
        test, remaining = split_fixed_test_stratified(torch.tensor(data["labels"]), seed=42)
        assert len(test) == claim["test_samples"]
        assert not set(test.tolist()) & set(remaining.tolist())
        metric_file = ROOT / "results" / cohort / "Ours/metrics_summary.csv"
        metrics = pd.read_csv(metric_file, index_col=0)
        per_run = pd.read_csv(metric_file.parent / "metrics_per_fold.csv")
        for name, key in [("Macro-AUC(OVR)", "macro_auc"), ("Macro-F1", "macro_f1")]:
            assert round(float(metrics.loc["Mean", name]), 4) == claim[key]
            np.testing.assert_allclose(per_run[name].mean(), metrics.loc["Mean", name], rtol=0, atol=1e-12)
        cohorts[cohort] = {"samples": len(data["labels"]), "species": dimensions,
                           "test_samples": len(test), "reported_metric_rounding": "matched"}
    report["checks"]["cohorts_and_metrics"] = cohorts
    for path in ROOT.rglob("*.py"):
        if ".git" not in path.parts:
            ast.parse(path.read_text(), filename=str(path))
    for path in [ROOT / "environment.yml", ROOT / "CITATION.cff"]:
        yaml.safe_load(path.read_text())
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert len(citation["authors"]) == 6 and citation["license"] == "MIT"
    report["checks"]["python_yaml_citation"] = "passed"
    # Check every archived result, including baseline/view records, against its frozen copy hash.
    sources = json.loads((ROOT / "docs/manuscript_source_manifest.json").read_text())
    for item in sources["sources"]:
        if "destination" in item:
            assert sha(ROOT / item["destination"]) == item["sha256"], item["destination"]
    for item in sources["derived_files"]:
        assert sha(ROOT / item["destination"]) == item["sha256"]
    summary = pd.read_csv(ROOT / "results/summary.csv")
    assert len(summary) == 45
    for row in summary.to_dict("records"):
        original = pd.read_csv(ROOT / row["source_file"], index_col=0)
        np.testing.assert_allclose(row["macro_auc"], original.loc["Mean", "Macro-AUC(OVR)"], rtol=0, atol=1e-12)
    report["checks"]["archived_results"] = {"settings": len(summary), "all_copy_hashes_match": True}
    # Validation-selection evaluation must predict exactly the validation indices, not test indices.
    rng = np.random.RandomState(8)
    data_path = work / "synthetic.npz"
    np.savez(data_path, X_otu_abundance=rng.uniform(0.01, 1, (40, 9)),
             X_text_embedding=rng.normal(size=(40, 13)), y=np.eye(4)[np.arange(40) % 4])
    config = TrainingConfig(epochs=1, repeats=2, patience=1, hd=8, k_neighbors=5, device="cpu")
    result = run_experiment(data_path, work, MetaDual, config, evaluation_split="validation")
    directory = Path(result["output_directory"])
    assert result["evaluation_split"] == "validation"
    for run in range(2):
        with np.load(directory / f"run_{run}/predictions.npz") as prediction, np.load(directory / f"run_{run}/splits.npz") as split:
            assert np.array_equal(prediction["indices"], split["validation"])
            assert not set(prediction["indices"]) & set(split["test"])
    report["checks"]["validation_scoring_indices"] = "passed for two trained runs"
    for cohort, dropout in [("primary", 0.5), ("independent", 0.55)]:
        output = work / "cli"
        completed = subprocess.run(
            [sys.executable, "-m", "model.train", "--cohort", cohort, "--data", str(data_path),
             "--output-dir", str(output), "--device", "cpu", "--epochs", "1", "--repeats", "1",
             "--patience", "1", "--hd", "8", "--k-neighbors", "5"], cwd=ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
            capture_output=True, text=True, timeout=120)
        assert completed.returncode == 0, completed.stderr
        saved = list((output / cohort / "full").glob("*/config.json"))
        assert len(saved) == 1
        settings = json.loads(saved[0].read_text())["config"]
        assert settings["dropout"] == dropout and settings["hd"] == 8 and settings["epochs"] == 1
    report["checks"]["cohort_cli_profiles_and_overrides"] = "both cohort CLIs trained synthetic data successfully"
    # Exercise Top 50 prompt construction and the evaluation-split dispatch without network access.
    import prompt_search.evaluation as evaluation
    from prompt_search.evaluation import EvaluationConfig, PromptEvaluator
    from prompt_search.search import SearchConfig, fuse_round
    from prompt_search.templates import ManualMicrobiomePrompter
    markdown = "### Functional Collaboration Groups\n" + "\n".join(
        f"- **Group {i}**: A, B\n**Function**: Mechanism {i}." for i in range(1, 11))
    captured = {}
    def create(**kwargs):
        captured["prompt"] = kwargs["messages"][1]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=markdown))])
    stub_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    class Embedder:
        def cls(self, texts):
            return np.ones((len(texts), 768), dtype=np.float32)
        def mean(self, texts, batch_size=2):
            captured["pair_batch_size"] = batch_size
            return np.eye(len(texts), dtype=np.float32)
    scorer = PromptEvaluator(work / "scorer", EvaluationConfig(device="cpu", top_k_species=50, score_split="validation"),
                             client=stub_client, embedder=Embedder())
    scorer.dataset = {"sample_ids": np.array(["sample"]), "otu_names": np.asarray([f"species_{i}" for i in range(277)]),
                      "X_otu_abundance": np.ones((1, 277)), "y": np.array([[1., 0.]])}
    responses = work / "responses"
    responses.mkdir()
    scorer._process_sample(0, "Input:\n{otu_text}", responses)
    assert len([line for line in captured["prompt"].splitlines() if line.startswith("- species_")]) == 50
    original_train = evaluation.run_experiment
    def stub_training(*args, **kwargs):
        captured["evaluation_split"] = kwargs["evaluation_split"]
        return {"pooled_macro_auc": 0.91, "mean_accuracy": 0.8}
    evaluation.run_experiment = stub_training
    try:
        assert scorer.evaluate("Input {otu_text}")["score"] == 0.91
        assert captured["evaluation_split"] == "validation"
    finally:
        evaluation.run_experiment = original_train
    records = [{"prompt_id": i, "prompt_hash": str(i), "prompt_text": f"Parent {i} {{otu_text}}", "score": 0.2}
               for i in range(3)]
    class Scorer:
        def evaluate(self, text):
            return {"prompt_hash": "pair-child", "prompt_text": text, "score": 0.8}
    pair, status = fuse_round(records, "pair_refinement", 2, 1, 3, 0.2, work / "pairs",
                              Scorer(), Embedder(), stub_client, ManualMicrobiomePrompter(), set())
    assert len(pair) == 1 and status["actual"] == 1 and captured["pair_batch_size"] == 2
    report["checks"]["manuscript_prompt_preset"] = "Top 50, validation split and two-parent dispatch passed with stubs"
    # Source files are read only if the original local workspace remains available.
    source_root = ROOT.parent
    if all((source_root / item["source"]).is_file() for item in sources["sources"]):
        for item in sources["sources"]:
            assert sha(source_root / item["source"]) == item["sha256"], item["source"]
        paper = source_root / "GMHI_GCN_SA_GitHub/MetaDual_Manuscript.docx"
        assert sha(paper) == manuscript["sha256"]
        source_npz = source_root / "data/final_multi_disease_all_features_768_one_2577_5_only_group_search_93.npz"
        with np.load(source_npz) as original, np.load(ROOT / "data_processing/data/independent_features.npz") as packed:
            assert original.files == packed.files
            for key in original.files:
                assert np.array_equal(original[key], packed[key]) and original[key].dtype == packed[key].dtype
        report["checks"]["new_sources_and_manuscript_unchanged"] = len(sources["sources"])
    else:
        report["checks"]["source_integrity"] = "skipped: source workspace not available"
    report["status"] = "passed"
    report["limits"] = ["No full-cohort retraining or paid API requests.",
                         "Historical Top 100/test-scoring discrepancies remain documented; no retrospective scores were invented."]
    (ROOT / "verification/manuscript_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
