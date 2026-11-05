"""The original 200-sample DeepSeek/FCG/BiomedBERT/GCN prompt scorer."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from data_processing.embedding import DEFAULT_EMBED_MODEL, BiomedicalEmbedder
from model.engine import TrainingConfig, run_experiment
from prompt_search.client import chat, make_client
from prompt_search.parsing import build_embedding_blocks, parse_fcgs_markdown
from prompt_search.scoring_network import GCNSA_two_graph


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EvaluationConfig:
    validation_csv: str = str(ROOT / "data_processing/data/prompt_validation.csv")
    embed_model: str = DEFAULT_EMBED_MODEL
    device: str = "cuda"
    max_workers: int = 20
    retries: int = 5
    task_temperature: float = 0.7
    epochs: int = 2000
    patience: int = 100
    repeats: int = 5
    top_k_species: int = 50
    score_split: str = "validation"


def validation_subset(csv_path):
    frame = pd.read_csv(csv_path, index_col=0, low_memory=False)
    otu_names = frame.index[:277].to_numpy()
    otu = frame.loc[otu_names].astype(float).fillna(0).to_numpy()
    sums = otu.sum(0, keepdims=True)
    abundance = np.divide(otu, sums, out=np.zeros_like(otu), where=sums > 0)
    labels = frame.loc["Label: Disease"].astype(str).str.strip().str.upper()
    diseases = labels.value_counts().head(4).index
    rng = np.random.RandomState(42)
    selected = []
    for disease in diseases:
        indices = np.flatnonzero(labels.to_numpy() == disease)
        selected.extend(rng.choice(indices, size=min(50, len(indices)), replace=False).tolist())
    selected_labels = labels.iloc[selected].to_numpy()
    categories, encoded = np.unique(selected_labels, return_inverse=True)
    info = []
    for key in ["Label: Age", "Label: BMI"]:
        values = pd.to_numeric(frame.loc[key], errors="coerce").fillna(0).to_numpy()[selected]
        info.append((values - values.min()) / (values.max() - values.min() + 1e-8))
    gender = frame.loc["Label: Gender"].astype(str).str.strip().str.lower().iloc[selected]
    info.append(gender.map({"male": 1.0, "female": 0.0}).fillna(0.5).to_numpy())
    return {"sample_ids": frame.columns.to_numpy()[selected].astype(str), "otu_names": otu_names.astype(str),
            "X_otu_abundance": abundance[:, selected].T, "X_info_raw": np.vstack(info).T,
            "y": np.eye(len(categories), dtype=np.float64)[encoded], "label_names": categories}


class PromptEvaluator:
    def __init__(self, output_dir, config=None, client=None, embedder=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or EvaluationConfig()
        self.client = client
        self.embedder = embedder
        self.dataset = validation_subset(self.config.validation_csv)

    def _process_sample(self, index, template, directory):
        values = self.dataset["X_otu_abundance"][index]
        # Stable sort preserves the original dictionary insertion order for tied abundances.
        top = np.argsort(-values, kind="stable")
        top = [i for i in top if values[i] > 0][:self.config.top_k_species]
        otu_text = "\n".join(f"- {self.dataset['otu_names'][i]}: {values[i]:.5f}" for i in top)
        if not otu_text:
            return None
        prompt = template.replace("{otu_text}", otu_text)
        sid = self.dataset["sample_ids"][index]
        for attempt in range(self.config.retries):
            attempt_id = time.time_ns()
            path = directory / f"sample_{index}_attempt_{attempt_id}.json"
            try:
                content = chat(self.client, prompt, self.config.task_temperature,
                               system="You are a bioinformatics assistant.")
                groups = parse_fcgs_markdown(content)
                path.write_text(json.dumps({"sample_id": sid, "prompt": prompt, "response": content,
                                            "parsed_groups": groups, "parse_success": len(groups) >= 10},
                                           ensure_ascii=False, indent=2) + "\n")
                if len(groups) >= 10:
                    return groups
            except Exception as exc:
                path.write_text(json.dumps({"sample_id": sid, "error_type": type(exc).__name__}, indent=2) + "\n")
                time.sleep(min(5 * (attempt + 1), 30))
        return None

    def evaluate(self, prompt_text):
        digest = hashlib.md5(prompt_text.encode()).hexdigest()[:8]
        directory = self.output_dir / digest
        result_file = directory / "evaluation.json"
        if result_file.exists():
            cached = json.loads(result_file.read_text())
            if cached["prompt_text"] != prompt_text:
                raise RuntimeError(f"Prompt short-hash collision: {digest}")
            return cached
        if "{otu_text}" not in prompt_text:
            return {"prompt_hash": digest, "prompt_text": prompt_text, "score": -1.0,
                    "status": "invalid", "reason": "Missing {otu_text} placeholder"}
        if self.client is None:
            self.client = make_client()
        response_dir = directory / "responses"
        response_dir.mkdir(parents=True, exist_ok=True)
        indices = list(range(len(self.dataset["sample_ids"])))
        groups_by_index = {}

        def process(indices_to_run):
            failed = []
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {executor.submit(self._process_sample, i, prompt_text, response_dir): i for i in indices_to_run}
                for future in as_completed(futures):
                    index = futures[future]
                    groups = future.result()
                    if groups:
                        groups_by_index[index] = groups
                    else:
                        failed.append(index)
            return failed

        failed = process(indices)
        first_round_failed = len(failed)
        if failed and len(failed) < 10:
            failed = process(failed)
        if failed:
            result = {"prompt_hash": digest, "prompt_text": prompt_text, "score": -1.0,
                      "status": "invalid", "first_round_failed": first_round_failed,
                      "failed_samples": [str(self.dataset["sample_ids"][i]) for i in failed]}
        else:
            if self.embedder is None:
                self.embedder = BiomedicalEmbedder(self.config.embed_model, self.config.device)
            embeddings, embedding_failures = [], []
            for index in indices:
                try:
                    blocks = build_embedding_blocks(groups_by_index[index][:10], batch_size=5)
                    vectors = self.embedder.cls(blocks)
                    if vectors.shape != (2, 768):
                        raise ValueError("Expected two 768-dimensional CLS vectors")
                    embeddings.append(vectors.reshape(-1))
                except Exception as exc:
                    embeddings.append(np.zeros(1536, dtype=np.float32))
                    embedding_failures.append({"sample_id": str(self.dataset["sample_ids"][index]),
                                               "error_type": type(exc).__name__})
            dataset = {key: value for key, value in self.dataset.items() if key != "label_names"}
            dataset["X_text_embedding"] = np.asarray(embeddings, dtype=np.float32)
            npz_path = directory / "features.npz"
            np.savez(npz_path, **dataset)
            profile = TrainingConfig(epochs=self.config.epochs, wd=1e-3, hd=64, dropout=0.55,
                                     patience=self.config.patience, repeats=self.config.repeats,
                                     k_neighbors=5, metric="cosine", device=self.config.device)
            # The original scorer did not reseed the CPU torch generator for each prompt.
            selection_options = {"evaluation_split": "validation"} if self.config.score_split == "validation" else {}
            training = run_experiment(npz_path, directory / "gcn", GCNSA_two_graph,
                                      profile, variant="prompt_scorer", seed_torch=False, **selection_options)
            score = training["pooled_macro_auc"]
            used_metric = f"{self.config.score_split}_pooled_macro_auc_ovr"
            if not math.isfinite(score):
                score = training["mean_accuracy"]
                used_metric = "mean_accuracy_fallback"
            result = {"prompt_hash": digest, "prompt_text": prompt_text, "score": float(score),
                      "status": "evaluated", "score_metric": used_metric,
                      "first_round_failed": first_round_failed, "embedding_failures": embedding_failures,
                      "training": training}
        result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result
