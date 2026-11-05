"""Dataset loading and the original symmetric k-nearest-neighbor graphs."""
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data_processing/data/final_features.npz"


def load_dataset(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    for key in ("X_otu_abundance", "X_text_embedding", "y"):
        if key not in result:
            raise ValueError(f"Missing NPZ field: {key}")
        if not np.isfinite(result[key]).all():
            raise ValueError(f"Non-finite values in {key}")
    n = len(result["y"])
    for key in ("X_otu_abundance", "X_text_embedding"):
        if result[key].ndim != 2 or len(result[key]) != n:
            raise ValueError(f"Invalid sample dimensions in {key}")
    y = result["y"]
    result["labels"] = y.argmax(1) if y.ndim == 2 else y.astype(np.int64)
    result.setdefault("sample_ids", np.arange(n).astype(str))
    if len(result["sample_ids"]) != n:
        raise ValueError("sample_ids does not match the feature rows")
    return result


def normalize_adj(matrix):
    rowsum = np.asarray(matrix.sum(1))
    with np.errstate(divide="ignore"):
        inverse = np.power(rowsum, -0.5).flatten()
    inverse[np.isinf(inverse)] = 0.0
    diagonal = sp.diags(inverse)
    return matrix.dot(diagonal).transpose().dot(diagonal)


def build_knn_graph(features, k=20, metric="braycurtis"):
    if not 2 <= k <= len(features):
        raise ValueError(f"k must be in [2, {len(features)}], received {k}")
    knn = NearestNeighbors(n_neighbors=k, metric=metric)
    knn.fit(features)
    _, indices = knn.kneighbors(features)
    adjacency = sp.lil_matrix((len(features), len(features)))
    for i in range(len(indices)):
        for j in range(1, k):
            adjacency[i, indices[i, j]] = 1
            adjacency[indices[i, j], i] = 1
    adjacency = normalize_adj(adjacency).tocsr()
    rows = np.repeat(np.arange(adjacency.shape[0]), np.diff(adjacency.indptr))
    tensor = torch.sparse_coo_tensor(
        torch.as_tensor(np.vstack([rows, adjacency.indices]), dtype=torch.long),
        torch.as_tensor(adjacency.data, dtype=torch.float32),
        size=adjacency.shape,
    )
    # Preserve the final maximum-with-transpose operation from both source loaders.
    dense = tensor.to_dense()
    dense = dense + dense.T * (dense.T > dense) - dense * (dense.T > dense)
    return dense.to_sparse().coalesce()
