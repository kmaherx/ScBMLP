from typing import Tuple, Collection, Optional

import numpy as np
from sklearn.model_selection import train_test_split
import scanpy as sc
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from .utils import get_cell_types, get_freqs


def _to_dense_numpy(X: np.ndarray) -> np.ndarray:
    """Convert dense/sparse matrix to dense numpy array."""
    if sp.issparse(X):
        return X.toarray()
    else:
        return X


def _encode_labels(labels: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Convert object array labels to integers and return mapping."""
    unique_labels = np.unique(labels)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_to_int[label] for label in labels])
    return encoded_labels, label_to_int


class ClassifierDataset(Dataset):
    """PyTorch dataset for single-cell classification tasks with memory-efficient indexing."""

    def __init__(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        label_mapping: dict,
        device: str = "cpu",
    ):
        self.device = device
        self.indices = indices
        self.label_mapping = label_mapping

        self.X = X  # dont index -> avoid copying; original X always on CPU
        self.y = labels[indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dataset_idx = self.indices[idx]

        x = torch.tensor(
            self.X[dataset_idx],
            dtype=torch.float32,
            device=self.device,
        )
        y = torch.tensor(self.y[idx], dtype=torch.long, device=self.device)
        return x, y

    def set_device(self, device: str):
        """Change the target device for tensor creation."""
        self.device = device


class RegressorDataset(Dataset):
    """PyTorch dataset for single-cell regression tasks with memory-efficient indexing."""

    def __init__(
        self,
        X: np.ndarray,
        y_data: np.ndarray,
        indices: np.ndarray,
        device: str = "cpu",
    ):
        self.device = device
        self.indices = indices

        self.X = X  # dont index -> avoid copying; original X always on CPU
        self.y = y_data[indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dataset_idx = self.indices[idx]

        x = torch.tensor(
            self.X[dataset_idx],
            dtype=torch.float32,
            device=self.device,
        )
        y = torch.tensor(self.y[idx], dtype=torch.float32, device=self.device)
        return x, y

    def set_device(self, device: str):
        """Change the target device for tensor creation."""
        self.device = device


def get_split_idxs(
    adata: sc.AnnData,
    train_split: float = 0.7,
    val_split: float = 0.15,
    random_state: int = 0,
    stratify_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) using two train_test_split calls.

    Contract:
        train fraction = train_split
        val fraction   = val_split
        test fraction  = 1 - train_split - val_split
        Requires: 0 < train_split < 1, 0 <= val_split < 1, train_split + val_split < 1.
        Stratifies if stratify_labels provided (classification use case).
    """
    if not (0 < train_split < 1 and 0 <= val_split < 1):
        raise ValueError("train_split must be in (0,1); val_split in [0,1).")
    test_split = 1.0 - train_split - val_split
    if test_split <= 0:
        raise ValueError("train_split + val_split must be < 1.")

    n = adata.shape[0]
    indices = np.arange(n)
    strat_all = stratify_labels if stratify_labels is not None else None

    train_idx, temp_idx, strat_train, strat_temp = train_test_split(
        indices,
        strat_all,
        test_size=(val_split + test_split),
        random_state=random_state,
        stratify=strat_all,
    )

    rel_test = test_split / (val_split + test_split)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=rel_test,
        random_state=random_state + 1,
        stratify=strat_temp if stratify_labels is not None else None,
    )

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def get_classification_datasets(
    adata: sc.AnnData,
    class_key: str,
    train_split: float = 0.7,
    val_split: float = 0.15,
    random_state: int = 0,
    device: str = "cpu",
) -> Tuple[ClassifierDataset, ClassifierDataset, ClassifierDataset]:
    """Create train/validation/test classification datasets.

    Args:
        train_split: Fraction for training set (default 0.7).
        val_split: Fraction for validation set (default 0.15). Test = 1 - train - val.
    """
    print("Creating dataset splits (stratified)...")
    all_labels = adata.obs[class_key].values
    train_indices, val_indices, test_indices = get_split_idxs(
        adata, train_split=train_split, val_split=val_split, random_state=random_state, stratify_labels=all_labels,
    )

    # Densify ONCE and share across all datasets
    print("Densifying data matrix (this may take a moment)...")
    X_dense = _to_dense_numpy(adata.X)

    print("Encoding labels...")
    encoded_labels, label_mapping = _encode_labels(all_labels)

    print("Creating dataset objects...")
    train_dataset = ClassifierDataset(X_dense, encoded_labels, train_indices, label_mapping, device)
    val_dataset = ClassifierDataset(X_dense, encoded_labels, val_indices, label_mapping, device)
    test_dataset = ClassifierDataset(X_dense, encoded_labels, test_indices, label_mapping, device)

    print(
        f"Datasets created - Train: {len(train_dataset):,}, "
        f"Val: {len(val_dataset):,}, Test: {len(test_dataset):,}"
    )
    return train_dataset, val_dataset, test_dataset


def get_regression_datasets(
    adata: sc.AnnData,
    target_key: str,
    train_split: float = 0.7,
    val_split: float = 0.15,
    random_state: int = 0,
    device: str = "cpu",
) -> Tuple[RegressorDataset, RegressorDataset, RegressorDataset]:
    """Create train/validation/test regression datasets.

    Args:
        train_split: Fraction for training set (default 0.7).
        val_split: Fraction for validation set (default 0.15). Test = 1 - train - val.
    """
    print("Creating dataset splits (shuffle)...")
    train_indices, val_indices, test_indices = get_split_idxs(
        adata, train_split=train_split, val_split=val_split, random_state=random_state,
    )

    print("Densifying data matrix (this may take a moment)...")
    X_dense = _to_dense_numpy(adata.X)
    y_data = adata.obsm[target_key]

    print("Creating dataset objects...")
    train_dataset = RegressorDataset(X_dense, y_data, train_indices, device)
    val_dataset = RegressorDataset(X_dense, y_data, val_indices, device)
    test_dataset = RegressorDataset(X_dense, y_data, test_indices, device)

    print(
        f"Datasets created - Train: {len(train_dataset):,}, "
        f"Val: {len(val_dataset):,}, Test: {len(test_dataset):,}"
    )
    return train_dataset, val_dataset, test_dataset


def myeloid_classes(
    n_cell_types: int = 3,
    class_key: str = "cell_type",
    train_split: float = 0.7,
    val_split: float = 0.15,
    normalize: bool = True,
    random_state: int = 0,
    device: str = "cpu",
) -> Tuple[sc.AnnData, ClassifierDataset, ClassifierDataset, ClassifierDataset]:
    """Paul15 myeloid development data for cell type classification."""
    adata = sc.datasets.paul15()

    if normalize:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata)

    get_cell_types(
        adata,
        n_comps=50,
        n_cell_types=n_cell_types,
        cell_type_key=class_key,
    )

    train_dataset, val_dataset, test_dataset = get_classification_datasets(
        adata,
        class_key,
    train_split=train_split,
        val_split=val_split,
        random_state=random_state,
        device=device,
    )

    return adata, train_dataset, val_dataset, test_dataset


def myeloid_freqs(
    k_neighbors: int = 15,
    n_freq_comps: int = 10,
    freq_key: str = "X_freq",
    train_split: float = 0.7,
    val_split: float = 0.15,
    normalize: bool = True,
    random_state: int = 0,
    device: str = "cpu",
) -> Tuple[sc.AnnData, RegressorDataset, RegressorDataset, RegressorDataset]:
    """Myeloid development with Laplacian eigenvector frequency targets."""
    adata = sc.datasets.paul15()

    if normalize:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata)

    get_freqs(adata, k=k_neighbors, n_freqs=n_freq_comps, device=device)

    train_dataset, val_dataset, test_dataset = get_regression_datasets(
        adata,
        target_key=freq_key,
    train_split=train_split,
        val_split=val_split,
        random_state=random_state,
        device=device,
    )

    return adata, train_dataset, val_dataset, test_dataset


def census_classes(
    census_config: dict,
    class_key: str,
    train_split: float = 0.7,
    val_split: float = 0.15,
    normalize: bool = True,
    random_state: int = 0,
    device: str = "cpu",
    census_version: str = "2025-01-30",
) -> Tuple[sc.AnnData, ClassifierDataset, ClassifierDataset, ClassifierDataset]:
    """
    Wrapper for retrieving and formatting a CELLxGENE Census query.

    Arguments:
        census_config: Configuration dictionary for the census query.
            - organism: Organism to filter the census data.
            - var_value_filter: Filter for desired genes based on their metadata.
            - obs_value_filter: Filter for desired cells based on their metadata.
            - var_column_names: Specify the gene metadata you want stored in the returned AnnData.
            - obs_column_names: Specify the cell metadata you want stored in the returned AnnData.
        class_key: Key for the target values for classification (e.g. "cell_type").
        census_version: Version of the census data to use.

        Example for sex classification of hepatocytes:
            ```python
            census_config = {
                "organism" : "Homo sapiens",
                "var_value_filter" : "feature_type in ['protein_coding']",
                "obs_value_filter" : "sex in ['male', 'female'] and cell_type == 'hepatocyte' and disease == 'normal'",
                "var_column_names" : ["feature_id", "feature_name", "feature_type", "feature_length"],
                "obs_column_names" : ["cell_type", "sex", "assay", "suspension_type"],
            }
            ```
        Can take several minutes depending on the query.
    """
    import cellxgene_census
    with cellxgene_census.open_soma(census_version=census_version) as census:
        adata = cellxgene_census.get_anndata(
            census=census,
            organism=census_config["organism"],
            var_value_filter=census_config["var_value_filter"],
            obs_value_filter=census_config["obs_value_filter"],
            var_column_names=census_config["var_column_names"],
            obs_column_names=census_config["obs_column_names"],
        )

        if normalize:
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            sc.pp.scale(adata)

        train_dataset, val_dataset, test_dataset = get_classification_datasets(
            adata, class_key, train_split=train_split, val_split=val_split, random_state=random_state, device=device,
        )

        return adata, train_dataset, val_dataset, test_dataset
