from .utils import get_cell_types, get_freqs

from typing import Tuple, Collection

from torch.utils.data import Dataset
import numpy as np
import scanpy as sc
import torch
import scipy.sparse as sp


def _to_dense_tensor(X: np.ndarray, device: str) -> torch.Tensor:
    """Convert dense/sparse matrix to dense tensor."""
    if sp.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = X
    return torch.tensor(X_dense, dtype=torch.float32).to(device)


def _encode_labels(labels: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Convert object array labels to integers and return mapping."""
    unique_labels = np.unique(labels)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_to_int[label] for label in labels])
    return encoded_labels, label_to_int


class ClassifierDataset(Dataset):
    """PyTorch dataset for single-cell classification tasks."""

    def __init__(
        self,
        adata: sc.AnnData,
        class_key: str,
        device: str = "cpu",
    ):
        self.adata = adata.copy()
        self.X = _to_dense_tensor(self.adata.X, device)

        labels = self.adata.obs[class_key].values
        encoded_labels, self.label_mapping = _encode_labels(labels)
        self.y = torch.tensor(encoded_labels, dtype=torch.long).to(device)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class RegressorDataset(Dataset):
    """PyTorch dataset for single-cell regression tasks."""
    
    def __init__(
        self,
        adata: sc.AnnData,
        freq_key: str = "X_freq",
        device: str = "cpu",
    ):
        self.adata = adata.copy()
        self.X = _to_dense_tensor(self.adata.X, device)
        self.y = torch.tensor(self.adata.obsm[freq_key], dtype=torch.float32).to(device)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_split_idxs(
    adata: sc.AnnData,
    val_split: float = 0.15,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset indices into train/validation/test sets."""
    np.random.seed(random_state)
    n = adata.shape[0]
    n_train = int(0.7 * n)
    n_val = int(val_split * n)
    n_test = n - n_train - n_val

    indices = np.random.permutation(n)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    return train_indices, val_indices, test_indices


def get_classification_datasets(
    adata: sc.AnnData,
    class_key: str,
    val_split: float = 0.15,
    random_state: int = 0,
    device: str = "cpu",
) -> Tuple[ClassifierDataset, ClassifierDataset, ClassifierDataset]:
    """Create train/validation/test classification datasets with automatic splitting."""
    train_indices, val_indices, test_indices = get_split_idxs(
        adata, val_split=val_split, random_state=random_state,
    )
    train_dataset = ClassifierDataset(adata[train_indices], class_key, device=device)
    val_dataset = ClassifierDataset(adata[val_indices], class_key, device=device)
    test_dataset = ClassifierDataset(adata[test_indices], class_key, device=device)
    return train_dataset, val_dataset, test_dataset


def get_regression_datasets(
    adata: sc.AnnData,
    freq_key: str = "X_freq",
    val_split: float = 0.15,
    random_state: int = 0,
    device: str = "cpu",
) -> Tuple[RegressorDataset, RegressorDataset, RegressorDataset]:
    """Create train/validation/test regression datasets with automatic splitting."""
    train_indices, val_indices, test_indices = get_split_idxs(
        adata, val_split=val_split, random_state=random_state,
    )
    train_dataset = RegressorDataset(adata[train_indices], freq_key, device=device)
    val_dataset = RegressorDataset(adata[val_indices], freq_key, device=device)
    test_dataset = RegressorDataset(adata[test_indices], freq_key, device=device)
    return train_dataset, val_dataset, test_dataset


def myeloid_classes(
    n_cell_types: int = 3,
    class_key: str = "cell_type",
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

    get_cell_types(adata, n_comps=50, n_cell_types=n_cell_types, cell_type_key=class_key)

    train_dataset, val_dataset, test_dataset = get_classification_datasets(
        adata, class_key, val_split=val_split, random_state=random_state, device=device,
    )

    return adata, train_dataset, val_dataset, test_dataset


def myeloid_freqs(
    k_neighbors: int = 15,
    n_freq_comps: int = 10,
    freq_key: str = "X_freq",
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
        adata, freq_key=freq_key, val_split=val_split, random_state=random_state, device=device,
    )

    return adata, train_dataset, val_dataset, test_dataset


def census_classes(
    census_config: dict,
    class_key: str,
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
            adata, class_key, val_split=val_split, random_state=random_state, device=device,
        )

        return adata, train_dataset, val_dataset, test_dataset
    