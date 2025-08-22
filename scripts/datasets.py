from .utils import get_cell_types, get_freqs

from typing import Tuple
from dataclasses import dataclass

from torch.utils.data import Dataset
import numpy as np
import scanpy as sc
import torch


class CellTypeDataset(Dataset):
    """PyTorch dataset for single-cell classification tasks."""
    def __init__(
        self,
        adata,
        indices: np.ndarray,
        label: str,
        device: str = "cpu"
    ):
        self.adata = adata[indices].copy()
        self.X = torch.tensor(self.adata.X, dtype=torch.float32).to(device)
        self.y = torch.tensor(self.adata.obs[label].values.astype(int)).to(device)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class CellFreqDataset(Dataset):
    """PyTorch dataset for single-cell frequency regression tasks."""
    def __init__(
        self,
        adata,
        indices: np.ndarray,
        freq_key: str = "X_freq",
        device: str = "cpu"
    ):
        self.adata = adata[indices].copy()
        self.X = torch.tensor(self.adata.X, dtype=torch.float32).to(device)
        self.y = torch.tensor(self.adata.obsm[freq_key], dtype=torch.float32).to(device)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_split_idxs(
    adata,
    val_split: float = 0.15,
    random_state: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset indices into train/validation/test sets."""
    n = adata.shape[0]
    n_train = int(0.7 * n)
    n_val = int(val_split * n)
    n_test = n - n_train - n_val

    indices = np.random.permutation(n)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    return train_indices, val_indices, test_indices


def get_type_datasets(
    adata,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    label: str,
    device: str = "cpu"
) -> Tuple[CellTypeDataset, CellTypeDataset, CellTypeDataset]:
    """Create train/val/test datasets for cell type classification."""
    train_dataset = CellTypeDataset(adata, train_indices, label, device=device)
    val_dataset = CellTypeDataset(adata, val_indices, label, device=device)
    test_dataset = CellTypeDataset(adata, test_indices, label, device=device)

    return train_dataset, val_dataset, test_dataset


def get_freq_datasets(
    adata,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    freq_key: str = "X_freq",
    device: str = "cpu"
) -> Tuple[CellFreqDataset, CellFreqDataset, CellFreqDataset]:
    """Create train/val/test datasets for frequency regression."""
    train_dataset = CellFreqDataset(adata, train_indices, freq_key, device=device)
    val_dataset = CellFreqDataset(adata, val_indices, freq_key, device=device)
    test_dataset = CellFreqDataset(adata, test_indices, freq_key, device=device)

    return train_dataset, val_dataset, test_dataset


def simulate(
    n_cells: int = 1000,
    n_genes: int = 100,
    n_cell_types: int = 5,
    cell_type_std: float = 1.0,
    val_split: float = 0.15,
    random_state: int = 0,
    device: str = "cpu",
):
    """Generate simulated single-cell data with Gaussian blobs for classification."""
    adata = sc.datasets.blobs(
        n_observations=n_cells,
        n_variables=n_genes,
        n_centers=n_cell_types,
        cluster_std=cell_type_std,
        random_state=random_state,
    )
    adata.obs["cell_type"] = adata.obs["blobs"]
    label = "cell_type"
    del adata.obs["blobs"]

    train_indices, val_indices, test_indices = get_split_idxs(
        adata, val_split=val_split, random_state=random_state
    )
    train_dataset, val_dataset, test_dataset = get_type_datasets(
        adata, train_indices, val_indices, test_indices, label, device=device
    )

    return adata, train_dataset, val_dataset, test_dataset


def myeloid_dev_type(
    n_cell_types: int = 3,
    val_split: float = 0.15,
    random_state: int = 0,
    device: str = "cpu",
):
    """Load and prepare Paul15 myeloid development data for cell type classification."""
    adata = sc.datasets.paul15()

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    get_cell_types(adata, n_comps=50, n_cell_types=n_cell_types)
    label = "cell_type"

    train_indices, val_indices, test_indices = get_split_idxs(
        adata, val_split=val_split, random_state=random_state
    )
    train_dataset, val_dataset, test_dataset = get_type_datasets(
        adata, train_indices, val_indices, test_indices, label, device=device
    )

    return adata, train_dataset, val_dataset, test_dataset


def myeloid_dev_freq(
    k_neighbors: int = 15,
    n_freq_comps: int = 10,
    val_split: float = 0.15,
    random_state: int = 0,
    device: str = "cpu",
):
    """
    Myeloid development dataset with Laplacian eigenvector-based frequency targets.
    Each cell is represented by Laplacian eigenvectors on the k-NN graph, providing
    a hierarchical frequency representation on the gene expression manifold.
    """
    adata = sc.datasets.paul15()

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    
    get_freqs(adata, k=k_neighbors, n_freqs=n_freq_comps, device=device)

    train_indices, val_indices, test_indices = get_split_idxs(
        adata, val_split=val_split, random_state=random_state
    )
    train_dataset, val_dataset, test_dataset = get_freq_datasets(
        adata, train_indices, val_indices, test_indices, freq_key="X_freq", device=device
    )

    return adata, train_dataset, val_dataset, test_dataset