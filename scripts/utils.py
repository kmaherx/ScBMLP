from typing import Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans


def get_cell_types(
    adata,
    n_comps: int = 50,
    n_cell_types: int = 3
) -> None:
    """Perform PCA-based cell type clustering on single-cell data."""
    _, V = np.linalg.eigh(adata.X.T @ adata.X)
    adata.obsm["X_pca"] = adata.X @ V[:, ::-1][:, :n_comps]

    clf = KMeans(n_clusters=n_cell_types, random_state=0)
    adata.obs["cell_type"] = clf.fit_predict(adata.obsm["X_pca"]).astype(str)


def calculate_knn_adjacency(
    X: torch.Tensor,
    k: int = 10,
    symmetric: bool = True
) -> torch.Tensor:
    """Calculate k-nearest neighbor indices for each cell based on cosine similarity."""
    n_cells = X.shape[0]
    X_norm = torch.nn.functional.normalize(X, p=2, dim=1)
    similarity = torch.mm(X_norm, X_norm.t())
    _, knn_indices = torch.topk(similarity, k + 1, dim=1)
    knn_indices = knn_indices[:, 1:]
    return knn_indices


def get_freqs(
    adata,
    k: int = 15,
    n_freqs: int = 50,
    device: str = "cpu"
) -> None:
    """Compute Laplacian eigenvector frequencies for hierarchical cell representation."""
    X_tensor = torch.tensor(adata.X, device=device, dtype=torch.float32)
    n_cells = X_tensor.shape[0]

    knn_indices = calculate_knn_adjacency(X_tensor, k=k)
    adata.obs["neighbors"] = [list(indices.cpu().numpy()) for indices in knn_indices]

    adj = torch.zeros(n_cells, n_cells, device=device)
    for i in range(n_cells):
        adj[i, knn_indices[i]] = 1.0
    adj = (adj + adj.t()) / 2
    adj = (adj > 0).float()

    laplacian = torch.diag(adj.sum(dim=0)) - adj
    vals, vecs = torch.linalg.eigh(laplacian)
    freq_vecs = vecs[:, 1:n_freqs+1].cpu().numpy()

    adata.uns["freq_vals"] = vals[1:n_freqs+1].cpu().numpy()
    adata.obsm["X_freq"] = freq_vecs
