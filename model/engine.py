"""Shared repeated-run training; model and graph profiles remain explicit."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from model.data import build_knn_graph, load_dataset
from model.splits import split_fixed_test_stratified, random_train_val_split_stratified


@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 3000
    lr: float = 0.02
    wd: float = 5e-4
    hd: int = 768
    dropout: float = 0.5
    epsilon: float = 0.9
    r: int = 3
    K: int = 3
    patience: int = 300
    repeats: int = 5
    k_neighbors: int = 20
    metric: str = "braycurtis"
    device: str = "cuda"


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu for a small smoke run.")
    return device


def save_checkpoint(state_dict, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(state_dict, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def auc_score(labels, probabilities):
    if probabilities.shape[1] == 2:
        return float(roc_auc_score(labels, probabilities[:, 1]))
    return float(roc_auc_score(labels, probabilities, average="macro", multi_class="ovr"))


def run_experiment(data_path, output_root, model_class, config, variant="full", seed_torch=True):
    if min(config.epochs, config.patience, config.repeats, config.K) < 1:
        raise ValueError("epochs, patience, repeats and K must be positive")
    if config.hd < 4 or config.hd % 4:
        raise ValueError("hd must be a positive multiple of four for the original attention layer")
    if seed_torch:
        torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device(config.device)
    data = load_dataset(data_path)
    n = len(data["labels"])
    if not 1 <= config.r <= n:
        raise ValueError("r must be between 1 and the sample count")
    labels_cpu = torch.as_tensor(data["labels"], dtype=torch.long)
    classes = np.unique(data["labels"])
    if not np.array_equal(classes, np.arange(len(classes))) or len(classes) < 2:
        raise ValueError("Labels must contain contiguous class indices starting at zero")
    idx_test_cpu, remaining = split_fixed_test_stratified(labels_cpu, test_ratio=0.2, seed=42)
    if not len(idx_test_cpu):
        raise ValueError("The fixed test set is empty; provide at least five samples per class")
    print(f"Building {config.metric} graphs: {n} samples, k={config.k_neighbors}", flush=True)
    adj1 = build_knn_graph(data["X_otu_abundance"], config.k_neighbors, config.metric).to(device)
    adj2 = build_knn_graph(data["X_text_embedding"], config.k_neighbors, config.metric).to(device)
    features_basic = torch.as_tensor(data["X_otu_abundance"], dtype=torch.float32, device=device)
    features_text = torch.as_tensor(data["X_text_embedding"], dtype=torch.float32, device=device)
    labels = labels_cpu.to(device)
    idx_test = idx_test_cpu.to(device)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = Path(output_root).resolve() / variant / timestamp
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {"variant": variant, "data": str(Path(data_path).resolve()),
                "model_class": model_class.__name__, "config": asdict(config),
                "fixed_test_seed": 42, "test_ratio": 0.2, "remaining_train_ratio": 0.75,
                "protocol": "Fixed stratified test set, repeated train/validation splits",
                "seed_torch_at_start": seed_torch}
    (directory / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows, perclass, matrices, predictions = [], [], [], []
    for run in range(config.repeats):
        run_dir = directory / f"run_{run}"
        run_dir.mkdir()
        train_cpu, val_cpu = random_train_val_split_stratified(remaining, labels_cpu, train_ratio=0.75, seed=run)
        if not len(train_cpu) or not len(val_cpu):
            raise ValueError("Training or validation split is empty")
        np.savez(run_dir / "splits.npz", train=train_cpu.numpy(), validation=val_cpu.numpy(), test=idx_test_cpu.numpy())
        train_idx, val_idx = train_cpu.to(device), val_cpu.to(device)
        network = model_class(nfeatures_basic=features_basic.shape[1], nfeatures_text=features_text.shape[1],
                              nhid=config.hd, nclass=len(classes), dropout1=config.dropout,
                              epsilon=config.epsilon, n_samples=n, r=config.r).to(device)
        # Source training scripts reset the CUDA dropout generator after model initialization.
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(config.seed)
        optimizer = torch.optim.Adam(network.parameters(), lr=config.lr, weight_decay=config.wd)
        best, bad, best_epoch = math.inf, 0, -1
        checkpoint = run_dir / "best.pt"
        history = []
        for epoch in range(config.epochs):
            network.train()
            optimizer.zero_grad()
            output, _, _ = network(features_basic, features_text, adj1, adj2, config.K)
            train_loss = F.nll_loss(output[train_idx], labels[train_idx])
            train_loss.backward()
            optimizer.step()
            network.eval()
            with torch.no_grad():
                output, _, _ = network(features_basic, features_text, adj1, adj2, config.K)
                val_loss = F.nll_loss(output[val_idx], labels[val_idx]).item()
                val_acc = (output[val_idx].argmax(1) == labels[val_idx]).float().mean().item()
            if not math.isfinite(val_loss):
                raise RuntimeError(f"Non-finite validation loss at run {run}, epoch {epoch}")
            history.append({"epoch": epoch, "train_loss": train_loss.item(), "validation_loss": val_loss,
                            "validation_accuracy": val_acc})
            if epoch % 10 == 0:
                print(f"run={run} epoch={epoch:04d} train_loss={train_loss.item():.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)
            if val_loss < best:
                save_checkpoint(network.state_dict(), checkpoint)
                best, best_epoch, bad = val_loss, epoch, 0
            else:
                bad += 1
            if bad >= config.patience:
                break
        network.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        network.eval()
        with torch.no_grad():
            output, _, _ = network(features_basic, features_text, adj1, adj2, config.K)
            probabilities = output[idx_test].exp().cpu().numpy()
        true = labels_cpu[idx_test_cpu].numpy()
        predicted = probabilities.argmax(1)
        report = classification_report(true, predicted, labels=classes, output_dict=True, zero_division=0)
        try:
            auc = auc_score(true, probabilities)
        except ValueError:
            auc = np.nan
        row = {"Run": run, "Accuracy": accuracy_score(true, predicted), "Macro-AUC(OVR)": auc}
        for average, prefix in [("macro avg", "Macro"), ("weighted avg", "Weighted")]:
            for metric, suffix in [("precision", "Precision"), ("recall", "Recall"), ("f1-score", "F1")]:
                row[f"{prefix}-{suffix}"] = report[average][metric]
        rows.append(row)
        for cls in classes:
            perclass.append({"Run": run, "Class": int(cls), **report[str(cls)]})
        matrix = confusion_matrix(true, predicted, labels=classes)
        matrices.append(matrix)
        np.savez_compressed(run_dir / "predictions.npz", indices=idx_test_cpu.numpy(),
                            sample_ids=data["sample_ids"][idx_test_cpu.numpy()],
                            labels=true, predictions=predicted, probabilities=probabilities)
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
        (run_dir / "metrics.json").write_text(json.dumps({**row, "best_epoch": best_epoch, "best_validation_loss": best}, indent=2) + "\n")
        predictions.append((true, predicted, probabilities))
        print(f"run={run} accuracy={row['Accuracy']:.4f} macro_auc={auc:.4f}", flush=True)
        del network, optimizer, output
    metrics = pd.DataFrame(rows)
    metrics.to_csv(directory / "metrics_per_run.csv", index=False)
    pd.DataFrame(perclass).to_csv(directory / "per_class_metrics_per_run.csv", index=False)
    metrics.drop(columns="Run").agg(["mean", "std"]).to_csv(directory / "metrics_summary.csv")
    np.save(directory / "confusion_matrices.npy", np.stack(matrices))
    combined_true = np.concatenate([p[0] for p in predictions])
    combined_predicted = np.concatenate([p[1] for p in predictions])
    combined_probs = np.concatenate([p[2] for p in predictions])
    np.savetxt(directory / "predictions.txt", combined_predicted, fmt="%d")
    np.savetxt(directory / "true_labels.txt", combined_true, fmt="%d")
    np.savetxt(directory / "sample_names.txt", np.tile(idx_test_cpu.numpy(), config.repeats), fmt="%d")
    np.savetxt(directory / "class_probs.txt", combined_probs, fmt="%.6f")
    try:
        combined_auc = auc_score(combined_true, combined_probs)
    except ValueError:
        combined_auc = float("nan")
    summary = {"output_directory": str(directory), "mean_accuracy": float(metrics.Accuracy.mean()),
               "accuracy_std_population": float(metrics.Accuracy.std(ddof=0)), "pooled_macro_auc": combined_auc}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
