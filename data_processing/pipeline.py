"""Prepare abundance features, generate FCGs, embed functions, and merge features."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from data_processing.embedding import DEFAULT_EMBED_MODEL, BiomedicalEmbedder
from prompt_search.parsing import parse_fcgs_markdown


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_processing/data"
GENERATED = ROOT / "outputs/data"


def exclusive_output(path, mode="w"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("xb" if "b" in mode else "x", **({} if "b" in mode else {"encoding": "utf-8"}))


def prepare(csv_path, output, min_class_count=100):
    df = pd.read_csv(csv_path, index_col=0)
    sample_ids = np.asarray(df.columns, dtype=str)
    labels = np.asarray([sample.rsplit("_", 1)[0] for sample in sample_ids])
    counts = pd.Series(labels).value_counts()
    keep = np.isin(labels, counts[counts >= min_class_count].index)
    abundance = df.to_numpy(dtype=float)[:, keep]
    if not np.isfinite(abundance).all() or (abundance < 0).any():
        raise ValueError("Abundances must be finite and non-negative")
    sums = abundance.sum(axis=0, keepdims=True)
    normalized = np.divide(abundance, sums, out=np.zeros_like(abundance), where=sums > 0)
    encoder = OneHotEncoder(sparse_output=False)
    one_hot = encoder.fit_transform(labels[keep].reshape(-1, 1))
    payload = {"sample_ids": np.asarray(sample_ids[keep].tolist(), dtype=str), "otu_names": np.asarray(df.index, dtype=str),
               "X_otu_abundance": normalized.T, "y": one_hot}
    with exclusive_output(output, "wb") as handle:
        np.savez(handle, **payload)
    mapping = [{"index": i, "label": str(label), "sample_count": int(one_hot[:, i].sum())}
               for i, label in enumerate(encoder.categories_[0])]
    with exclusive_output(Path(output).with_suffix(".labels.json")) as handle:
        json.dump(mapping, handle, indent=2)
    return payload


def generate_fcgs(csv_path, output, prompt_path, top_k=100, max_workers=20, retries=5):
    from prompt_search.client import chat, make_client
    client = make_client()
    template = Path(prompt_path).read_text()
    if "{otu_text}" not in template:
        raise ValueError("The prompt template must contain {otu_text}")
    frame = pd.read_csv(csv_path, index_col=0).apply(pd.to_numeric, errors="coerce").fillna(0)
    frame = frame.div(frame.sum(axis=0), axis=1).fillna(0)

    def process(sample_id):
        values = frame[sample_id]
        top = values[values > 0].sort_values(ascending=False).head(top_k)
        otu_text = "\n".join(f"- {otu}: {value:.5f}" for otu, value in top.items())
        if not otu_text:
            return sample_id, None
        last_error = "No functional groups parsed"
        for attempt in range(retries):
            try:
                content = chat(client, template.replace("{otu_text}", otu_text), 0.2,
                               system="You are a bioinformatics assistant.")
                groups = parse_fcgs_markdown(content)
                if groups:
                    return sample_id, {"sample_id": sample_id,
                                       "parsed_response": {"Functional_Collaboration_Groups": groups}}
            except Exception as exc:
                last_error = type(exc).__name__
                time.sleep(5 * (attempt + 1))
        return sample_id, {"error": last_error}

    failed = []
    with exclusive_output(output) as handle, ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process, sample_id) for sample_id in frame.columns]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating FCGs"):
            sample_id, record = future.result()
            if not record or "error" in record:
                failed.append(sample_id)
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    if failed:
        with exclusive_output(Path(output).with_suffix(".failed.json")) as handle:
            json.dump(failed, handle, indent=2)
        raise RuntimeError(f"FCG generation failed for {len(failed)} samples; see the failed-sample file")


def embed_fcgs(input_path, output, model_path=DEFAULT_EMBED_MODEL, device="cpu"):
    embedder = BiomedicalEmbedder(model_path, device)
    with Path(input_path).open() as source, exclusive_output(output) as target:
        for line in tqdm(source, desc="Embedding function batches"):
            record = json.loads(line)
            parsed = record.get("parsed_response", {})
            groups = parsed.get("Functional_Collaboration_Groups", [])
            valid = [(group.get("Function", "").strip(), group.get("OTUs", []))
                     for group in groups if group.get("Function", "").strip()]
            embedded = []
            for start in range(0, len(valid), 5):
                batch = valid[start:start + 5]
                text = " ".join(function for function, _ in batch)
                vector = embedder.cls([text])[0]
                if vector.shape != (768,):
                    raise ValueError("The selected embedder must produce 768-dimensional vectors")
                embedded.append({"Function_Batch": text, "OTUs": [otus for _, otus in batch],
                                 "Embedding": vector.tolist()})
            record["functional_group_embeddings"] = embedded
            parsed.pop("Functional_Collaboration_Groups", None)
            target.write(json.dumps(record, ensure_ascii=False) + "\n")


def merge_features(base_path, embeddings_path, output):
    embeddings = {}
    with Path(embeddings_path).open() as handle:
        for line in handle:
            record = json.loads(line)
            groups = record.get("functional_group_embeddings", [])
            if groups:
                embeddings[record["sample_id"]] = [group["Embedding"] for group in groups]
    with np.load(base_path, allow_pickle=False) as handle:
        result = {key: handle[key] for key in handle.files}
    rows, missing = [], []
    for sid in result["sample_ids"]:
        groups = embeddings.get(str(sid), [])
        vectors = []
        if not groups:
            missing.append(str(sid))
        for i in range(2):
            vector = np.asarray(groups[i] if i < len(groups) else np.zeros(768), dtype=np.float32)
            if vector.shape != (768,) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid embedding for sample {sid}")
            vectors.append(vector)
        rows.append(np.concatenate(vectors))
    result["X_text_embedding"] = np.vstack(rows)
    with exclusive_output(output, "wb") as handle:
        np.savez(handle, **result)
    with exclusive_output(Path(output).with_suffix(".merge_report.json")) as handle:
        json.dump({"samples": len(rows), "text_dimensions": 1536, "missing_embedding_samples": missing}, handle, indent=2)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Normalize abundance and filter labels")
    prepare_parser.add_argument("--input", type=Path, default=DATA / "4347_final_relative_abundances.csv")
    prepare_parser.add_argument("--output", type=Path, default=GENERATED / "base_features.npz")
    prepare_parser.add_argument("--min-class-count", type=int, default=100)
    generate_parser = commands.add_parser("generate", help="Online DeepSeek generation; requires API credentials")
    generate_parser.add_argument("--input", type=Path, default=DATA / "4347_final_relative_abundances.csv")
    generate_parser.add_argument("--output", type=Path, default=GENERATED / "fcg_results.jsonl")
    generate_parser.add_argument("--prompt", type=Path, default=ROOT / "prompt_search/prompts/data_generation_prompt.txt")
    generate_parser.add_argument("--top-k", type=int, default=100)
    generate_parser.add_argument("--max-workers", type=int, default=20)
    generate_parser.add_argument("--retries", type=int, default=5)
    embed_parser = commands.add_parser("embed", help="Embed archived or newly generated FCG functions")
    embed_parser.add_argument("--input", type=Path, default=DATA / "fcg_results.jsonl")
    embed_parser.add_argument("--output", type=Path, default=GENERATED / "embeddings.jsonl")
    embed_parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    embed_parser.add_argument("--device", default="cpu")
    merge_parser = commands.add_parser("merge", help="Join the first two embeddings to abundance features")
    merge_parser.add_argument("--base", type=Path, default=GENERATED / "base_features.npz")
    merge_parser.add_argument("--embeddings", type=Path, default=GENERATED / "embeddings.jsonl")
    merge_parser.add_argument("--output", type=Path, default=GENERATED / "final_features.npz")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.input, args.output, args.min_class_count)
        print(f"Saved {len(result['sample_ids'])} samples to {args.output}")
    elif args.command == "generate":
        generate_fcgs(args.input, args.output, args.prompt, args.top_k, args.max_workers, args.retries)
    elif args.command == "embed":
        embed_fcgs(args.input, args.output, args.embed_model, args.device)
    else:
        merge_features(args.base, args.embeddings, args.output)


if __name__ == "__main__":
    main()
