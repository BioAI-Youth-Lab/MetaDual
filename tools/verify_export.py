"""Offline export verification. Writes only within this exported project."""
import argparse
import ast
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from contextlib import redirect_stdout

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_data(work, source_root=None):
    from data_processing.pipeline import prepare, merge_features
    from model.data import DEFAULT_DATA, load_dataset
    data = load_dataset(DEFAULT_DATA)
    expected = {"X_otu_abundance": (4012, 903), "X_text_embedding": (4012, 1536), "y": (4012, 7)}
    for key, shape in expected.items():
        assert data[key].shape == shape, (key, data[key].shape)
        assert np.isfinite(data[key]).all()
    assert len(set(data["sample_ids"])) == 4012
    assert np.allclose(data["X_otu_abundance"].sum(1), 1)
    counts = data["y"].sum(0).astype(int).tolist()
    assert counts == [152, 298, 108, 2636, 342, 238, 238]
    base_path = work / "base_features.npz"
    prepared = prepare(ROOT / "data_processing/data/4347_final_relative_abundances.csv", base_path)
    for key in prepared:
        assert np.array_equal(prepared[key], data[key]), key
        assert prepared[key].dtype == data[key].dtype, key
    try:
        prepare(ROOT / "data_processing/data/4347_final_relative_abundances.csv", base_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("prepare overwrote its output")
    fcg_records = {}
    with (ROOT / "data_processing/data/fcg_results.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            fcg_records[record["sample_id"]] = record
    assert len(fcg_records) == 4347
    assert all(len(r["parsed_response"]["Functional_Collaboration_Groups"]) >= 10 for r in fcg_records.values())
    assert set(data["sample_ids"]) <= set(fcg_records)
    details = {"shape": {key: list(shape) for key, shape in expected.items()},
               "class_counts": counts, "abundance_and_labels": "array-identical reconstruction",
               "fcg_records": len(fcg_records)}
    if source_root is not None:
        embeddings = source_root / "data_gmhi/deepseek_top100_network_analysis_embeddings_group_5_4347.jsonl"
        merged = merge_features(base_path, embeddings, work / "merged_features.npz")
        for key in prepared:
            assert np.array_equal(merged[key], data[key]), key
        assert np.array_equal(merged["X_text_embedding"], data["X_text_embedding"])
        details["full_embedding_merge"] = "all 4012 rows array-identical to supplied final NPZ"
        model_dir = source_root / "deepseek/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        if model_dir.is_dir():
            from data_processing.embedding import BiomedicalEmbedder
            embedder = BiomedicalEmbedder(str(model_dir), "cpu")
            errors = []
            for index in [0, len(data["sample_ids"]) // 2, len(data["sample_ids"]) - 1]:
                sid = data["sample_ids"][index]
                groups = fcg_records[sid]["parsed_response"]["Functional_Collaboration_Groups"]
                functions = [g["Function"].strip() for g in groups if g["Function"].strip()]
                blocks = [" ".join(functions[start:start + 5]) for start in [0, 5]]
                embedded = embedder.cls(blocks).reshape(-1)
                delta = float(np.max(np.abs(embedded - data["X_text_embedding"][index])))
                np.testing.assert_allclose(embedded, data["X_text_embedding"][index], atol=2e-5, rtol=2e-5)
                errors.append(delta)
            details["representative_fresh_embeddings"] = {"samples": 3, "max_absolute_errors": errors}
        else:
            details["representative_fresh_embeddings"] = "skipped: original text-model directory unavailable"
    else:
        details["source_embedding_checks"] = "skipped: no --source-root supplied"
    return details


def load_definitions(path, names, namespace):
    source = path.read_text()
    available = {node.name: node for node in ast.parse(source).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    module = ast.Module(body=[available[name] for name in names], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)


def compare_sources(source_root):
    import scipy.sparse as sp
    from sklearn.neighbors import NearestNeighbors
    from torch.nn import functional as F
    from model import layers
    from model.data import build_knn_graph
    from model.networks import VARIANTS
    from prompt_search.scoring_network import GCNSA_two_graph as ScoringNetwork
    from model.similarity import consinesimilar
    main_dir = source_root / "GCN_version_nature_data_gmhi/GCN-SA-main/GCN-SA/pygcn"
    prompt_dir = source_root / "comp/got/prompt1/GCN-SA/pygcn"
    rng = np.random.RandomState(21)
    inputs = rng.uniform(0.01, 1, size=(24, 9))
    graphs = {}
    for name, directory, metric in [("main", main_dir, "braycurtis"), ("prompt", prompt_dir, "cosine")]:
        namespace = {"np": np, "sp": sp, "th": torch, "NearestNeighbors": NearestNeighbors}
        load_definitions(directory / "utils_data.py", ["normalize_adj", "build_knn_graph"], namespace)
        original = namespace["build_knn_graph"](inputs, 5).to_dense()
        cleaned = build_knn_graph(inputs, 5, metric).to_dense()
        torch.testing.assert_close(original, cleaned, atol=0, rtol=0)
        graphs[name] = "exact numerical equality"
    if not torch.cuda.is_available():
        return {"graph_comparisons": graphs, "network_comparisons": "skipped: CUDA required by original definitions"}
    device = torch.device("cuda:0")
    source_models = {}
    for name, directory, class_names in [
        ("main", main_dir, [cls.__name__ for cls in VARIANTS.values()]),
        ("prompt", prompt_dir, ["GCNSA_two_graph"]),
    ]:
        namespace = {"torch": torch, "nn": torch.nn, "F": F, "np": np, "math": math,
                     "Parameter": torch.nn.Parameter, "Module": torch.nn.Module,
                     "consinesimilar": consinesimilar}
        load_definitions(directory / "layers.py", ["dotproduct", "adjcentmatrix", "MultiHeadAttention", "EncoderLayer1", "normalize_adj"], namespace)
        load_definitions(directory / "models.py", class_names, namespace)
        source_models[name] = namespace
    args = {"nfeatures_basic": 9, "nfeatures_text": 13, "nhid": 8, "nclass": 4,
            "dropout1": 0.2, "epsilon": 0.9, "n_samples": 24, "r": 3}
    x1 = torch.tensor(rng.uniform(0.01, 1, size=(24, 9)), dtype=torch.float32, device=device)
    x2 = torch.tensor(rng.normal(size=(24, 13)), dtype=torch.float32, device=device)
    adjacency = torch.eye(24, device=device).to_sparse()
    labels = torch.arange(24, device=device) % 4
    results = {}
    for name, cleaned_class, namespace in [(name, cls, source_models["main"]) for name, cls in VARIANTS.items()] + [("prompt_scorer", ScoringNetwork, source_models["prompt"])]:
        original = namespace[cleaned_class.__name__](**args).to(device)
        cleaned = cleaned_class(**args).to(device)
        cleaned.load_state_dict(original.state_dict(), strict=True)
        original.eval()
        cleaned.eval()
        with torch.no_grad():
            expected = original(x1, x2, adjacency, adjacency, 3)[0]
            actual = cleaned(x1, x2, adjacency, adjacency, 3)[0]
        torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)
        original.train()
        cleaned.train()
        torch.cuda.manual_seed(123)
        before = original(x1, x2, adjacency, adjacency, 3)[0]
        F.nll_loss(before, labels).backward()
        torch.cuda.manual_seed(123)
        after = cleaned(x1, x2, adjacency, adjacency, 3)[0]
        F.nll_loss(after, labels).backward()
        torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)
        gradient_error = 0.0
        for (parameter_name, p), (other_name, q) in zip(original.named_parameters(), cleaned.named_parameters()):
            assert parameter_name == other_name
            assert (p.grad is None) == (q.grad is None), parameter_name
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)
                gradient_error = max(gradient_error, float((p.grad - q.grad).abs().max().item()))
        results[name] = {"eval_max_absolute_error": float((expected - actual).abs().max().item()),
                         "train_max_absolute_error": float((before - after).abs().max().item()),
                         "gradient_max_absolute_error": gradient_error,
                         "state_dict_keys": len(original.state_dict())}
    return {"graph_comparisons": graphs, "network_comparisons": results}


class StubClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(self.responses)))])


def check_prompts(work):
    from prompt_search.history import load_history
    from prompt_search.search import SearchConfig, generate_initial, evaluate_initial, fuse_round
    from prompt_search.templates import ManualMicrobiomePrompter
    from prompt_search.parsing import parse_fcgs_markdown
    from prompt_search.evaluation import validation_subset, PromptEvaluator, EvaluationConfig
    import prompt_search.evaluation as evaluation
    stages = load_history()
    assert [len(stages[stage]) for stage in ["initial", "round1", "round2"]] == [30, 15, 8]
    best = max((r for rows in stages.values() for r in rows), key=lambda r: r["score"])
    assert (ROOT / "prompt_search/prompts/best_prompt.txt").read_text().rstrip() == best["prompt_text"].rstrip()
    assert best["prompt_hash"] == "168f2812"
    template = ManualMicrobiomePrompter()
    assert "{otu_text}" in template.generate_prompt_template()
    initial_client = StubClient(["A {otu_text}", "A {otu_text}", "B {otu_text}"])
    candidates, seen = generate_initial(initial_client, template, work / "prompt_mock", SearchConfig(initial_count=3))
    assert len(candidates) == 2 and initial_client.calls == 3
    class StubEvaluator:
        def evaluate(self, text):
            return {"prompt_text": text, "prompt_hash": hashlib.md5(text.encode()).hexdigest()[:8], "score": 0.8}
    evaluated = evaluate_initial(candidates, StubEvaluator(), work / "prompt_mock")
    assert len(evaluated) == 2
    parents = [{"prompt_id": i + 1, "prompt_text": f"parent {i} {{otu_text}}",
                "prompt_hash": str(i), "score": 0.1 * (i + 1)} for i in range(5)]
    class StubEmbedder:
        def mean(self, texts, batch_size=4):
            return np.random.RandomState(2).normal(size=(len(texts), 8))
        def cls(self, texts):
            return np.ones((len(texts), 768), dtype=np.float32)
    fused, status = fuse_round(parents, "round1", 4, 2, 3, 1.0, work / "prompt_mock", StubEvaluator(),
                               StubEmbedder(), StubClient(["fused {otu_text}"] * 3), template, set())
    assert len(fused) == 1 and status["status"] == "partial" and status["actual"] == 1
    empty, empty_status = fuse_round(parents[:2], "round2", 3, 8, 70, 0.5, work / "prompt_mock", StubEvaluator(),
                                     StubEmbedder(), StubClient([]), template, set())
    assert empty == [] and empty_status["status"] == "insufficient_valid_parents"
    subset = validation_subset(ROOT / "data_processing/data/prompt_validation.csv")
    assert subset["X_otu_abundance"].shape == (200, 277)
    assert subset["y"].sum(0).tolist() == [50] * 4
    markdown = "### Functional Collaboration Groups\n" + "\n".join(
        f"- **Group {i}**: OTU_A, OTU_B\n**Function**: Function {i}." for i in range(1, 11))
    assert len(parse_fcgs_markdown(markdown)) == 10
    api = StubClient([markdown] * 200)
    captured = {}
    original_train = evaluation.run_experiment
    def mock_training(path, output, model_class, config, variant, seed_torch):
        captured.update(asdict(config))
        assert not seed_torch and variant == "prompt_scorer"
        with np.load(path) as data:
            assert data["X_text_embedding"].shape == (200, 1536)
        return {"pooled_macro_auc": 0.9, "mean_accuracy": 0.8}
    evaluation.run_experiment = mock_training
    try:
        evaluator = PromptEvaluator(work / "scorer_mock", EvaluationConfig(device="cpu"), client=api, embedder=StubEmbedder())
        scored = evaluator.evaluate("Test {otu_text}")
        cached = evaluator.evaluate("Test {otu_text}")
        assert scored["score"] == cached["score"] == 0.9 and api.calls == 200
        invalid = evaluator.evaluate("Missing placeholder")
        assert invalid["score"] == -1.0 and api.calls == 200
        assert captured["metric"] == "cosine" and captured["k_neighbors"] == 5 and captured["hd"] == 64
    finally:
        evaluation.run_experiment = original_train
    return {"counts": [30, 15, 8], "best_hash": best["prompt_hash"],
            "mocked_checks": ["deduplication without refill", "partial aggregation", "insufficient parents",
                               "200-sample FCG/embedding/scorer integration", "per-run evaluation cache", "invalid placeholder"],
            "external_api_calls": 0}


def check_training(work):
    from model.engine import run_experiment, TrainingConfig
    from model.networks import VARIANTS
    from prompt_search.scoring_network import GCNSA_two_graph
    rng = np.random.RandomState(3)
    path = work / "synthetic.npz"
    np.savez(path, X_otu_abundance=rng.uniform(0.01, 1, size=(40, 9)),
             X_text_embedding=rng.normal(size=(40, 13)).astype(np.float32),
             y=np.eye(4)[np.arange(40) % 4], sample_ids=np.asarray([f"synthetic_{i}" for i in range(40)]))
    checks = {}
    for variant, cls in list(VARIANTS.items()) + [("prompt_scorer", GCNSA_two_graph)]:
        config = TrainingConfig(epochs=1, patience=1, repeats=1, hd=8, k_neighbors=5, device="cpu",
                                metric="cosine" if variant == "prompt_scorer" else "braycurtis")
        with redirect_stdout(io.StringIO()) as log:
            result = run_experiment(path, work / "training", cls, config, variant)
        directory = Path(result["output_directory"])
        assert (directory / "run_0/best.pt").is_file()
        with np.load(directory / "run_0/splits.npz") as split:
            groups = [set(split[key].tolist()) for key in ["train", "validation", "test"]]
            assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            assert len(set.union(*groups)) == 40
        with np.load(directory / "run_0/predictions.npz") as prediction:
            assert prediction["probabilities"].shape == (8, 4)
            assert np.isfinite(prediction["probabilities"]).all()
            np.testing.assert_allclose(prediction["probabilities"].sum(1), 1, atol=1e-6)
        checks[variant] = {"epochs": 1, "samples": 40, "device": "cpu", "status": "passed"}
    return checks


def check_notebook_and_cli(work):
    import nbformat
    from nbclient import NotebookClient
    notebook_path = ROOT / "prompt_search/manual_prompt_search_agg_no.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
            ast.parse(cell.source)
    client = NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}},
                            allow_errors=False)
    client.execute(env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    nbformat.write(notebook, work / "executed_offline_notebook.ipynb")
    for module, args in [("model.train", ["--help"]), ("data_processing.pipeline", ["--help"]),
                         ("prompt_search.search", ["--help"]), ("prompt_search.search", ["history"])]:
        completed = subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT,
                                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                                   capture_output=True, text=True, timeout=120)
        assert completed.returncode == 0, completed.stderr
    subprocess.run(["bash", "-n", str(ROOT / "model/run_all.sh")], check=True)
    return {"notebook_cells": len(notebook.cells), "offline_notebook_execution": "passed",
            "command_help_and_history": "passed", "batch_shell_syntax": "passed"}


def check_security_and_files():
    secret_pattern = re.compile(r"sk-[A-Za-z0-9_-]{16,}")
    forbidden = []
    big = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or "work" in path.relative_to(ROOT).parts:
            continue
        if path.stat().st_size > 100 * 1024 * 1024:
            big.append(str(path.relative_to(ROOT)))
        if path.suffix in [".py", ".ipynb", ".json", ".md", ".txt", ".yml", ".sh", ".csv", ".jsonl"]:
            content = path.read_text(errors="replace")
            if secret_pattern.search(content):
                forbidden.append(str(path.relative_to(ROOT)))
            if path.suffix in [".py", ".ipynb", ".sh"]:
                machine_prefix = "/home/" + "data/" + "yww/"
                assert machine_prefix not in content, f"Hardcoded machine path in {path}"
    assert not forbidden, f"Credential-like tokens in {forbidden}"
    assert not big, f"Unexpected large files: {big}"
    return {"credential_scan": "passed", "hardcoded_machine_path_scan": "passed", "files_over_100_MiB": big}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    work = ROOT / "verification/work" / stamp
    work.mkdir(parents=True, exist_ok=False)
    report = {"started_utc": stamp, "python": sys.version.split()[0], "torch": torch.__version__,
              "cuda_build": torch.version.cuda, "external_api_calls": 0, "checks": {}}
    copied = json.loads((ROOT / "docs/copied_files.json").read_text())
    for item in copied:
        assert digest(ROOT / item["destination"]) == item["sha256"], item["destination"]
    report["checks"]["copied_files"] = {"count": len(copied), "status": "all hashes match"}
    print("Copied-file hashes passed", flush=True)
    report["checks"]["data"] = check_data(work, args.source_root)
    print("Data reconstruction passed", flush=True)
    report["checks"]["prompts"] = check_prompts(work)
    print("Prompt archive and offline orchestration passed", flush=True)
    report["checks"]["training"] = check_training(work)
    print("Five model CPU smoke runs passed", flush=True)
    if args.source_root:
        report["checks"]["original_numerical_comparison"] = compare_sources(args.source_root)
        manifest_path = ROOT / "docs/source_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["files"]:
            item["sha256_after"] = digest(args.source_root / item["source"])
            item["unchanged"] = item["sha256_before"] == item["sha256_after"]
            assert item["unchanged"], item["source"]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        report["checks"]["source_integrity"] = {"files": len(manifest["files"]), "all_unchanged": True}
        print("Original numerical comparison and source integrity passed", flush=True)
    else:
        report["checks"]["source_comparison"] = "skipped: no --source-root supplied"
    report["checks"]["interfaces"] = check_notebook_and_cli(work)
    print("Notebook and command interfaces passed", flush=True)
    report["checks"]["export_scan"] = check_security_and_files()
    report["status"] = "passed"
    report["limits"] = ["No full-length training or paid API generation was run.",
                         "Fresh text embeddings were checked on representative samples only when local model files were available."]
    (ROOT / "verification/report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
