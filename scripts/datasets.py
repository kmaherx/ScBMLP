from .utils import get_cell_types, get_freqs

from typing import Tuple, Collection

from torch.utils.data import Dataset
import numpy as np
import scanpy as sc
import torch
import scipy.sparse as sp


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
        X: np.ndarray,  # Pre-densified shared data
        labels: np.ndarray,  # Pre-processed labels
        indices: np.ndarray,
        label_mapping: dict,
        device: str = "cpu",
    ):
        self.X = X  # Shared dense matrix (always on CPU)
        self.device = device  # Target device for tensors
        self.indices = indices
        self.label_mapping = label_mapping
        
        # Only keep labels for our subset (always on CPU)
        self.y = labels[indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Map dataset index to original data index
        actual_idx = self.indices[idx]
        
        # Convert to tensors and transfer to target device only when requested
        x = torch.tensor(self.X[actual_idx], dtype=torch.float32, device=self.device)
        y = torch.tensor(self.y[idx], dtype=torch.long, device=self.device)
        return x, y
    
    def set_device(self, device: str):
        """Change the target device for tensor creation."""
        self.device = device


class RegressorDataset(Dataset):
    """PyTorch dataset for single-cell regression tasks with memory-efficient indexing."""
    
    def __init__(
        self,
        X: np.ndarray,  # Pre-densified shared data
        y_data: np.ndarray,  # Pre-processed target data
        indices: np.ndarray,
        device: str = "cpu",
    ):
        self.X = X  # Shared dense matrix (always on CPU)
        self.device = device  # Target device for tensors
        self.indices = indices
        
        # Only keep targets for our subset (always on CPU)
        self.y = y_data[indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Map dataset index to original data index
        actual_idx = self.indices[idx]
        
        # Convert to tensors and transfer to target device only when requested
        x = torch.tensor(self.X[actual_idx], dtype=torch.float32, device=self.device)
        y = torch.tensor(self.y[idx], dtype=torch.float32, device=self.device)
        return x, y
    
    def set_device(self, device: str):
        """Change the target device for tensor creation."""
        self.device = device


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
    """Create train/validation/test classification datasets with memory-efficient splitting."""
    print("Creating dataset splits...")
    train_indices, val_indices, test_indices = get_split_idxs(
        adata, val_split=val_split, random_state=random_state,
    )
    
    print("Densifying data matrix (this may take a moment)...")
    # Densify ONCE and share across all datasets
    X_dense = _to_dense_numpy(adata.X)
    
    print("Encoding labels...")
    # Process labels ONCE
    all_labels = adata.obs[class_key].values
    encoded_labels, label_mapping = _encode_labels(all_labels)
    
    print("Creating dataset objects...")
    # Create datasets that share the same dense matrix
    train_dataset = ClassifierDataset(X_dense, encoded_labels, train_indices, label_mapping, device)
    val_dataset = ClassifierDataset(X_dense, encoded_labels, val_indices, label_mapping, device)
    test_dataset = ClassifierDataset(X_dense, encoded_labels, test_indices, label_mapping, device)
    
    # Store adata reference for compatibility with existing code
    train_dataset.adata = adata
    val_dataset.adata = adata
    test_dataset.adata = adata
    
    print(f"Datasets created - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}, Test: {len(test_dataset):,}")
    return train_dataset, val_dataset, test_dataset


def get_regression_datasets(
    adata: sc.AnnData,
    target_key: str,
    val_split: float = 0.15,
    random_state: int = 0,
    device: str = "cpu",
) -> Tuple[RegressorDataset, RegressorDataset, RegressorDataset]:
    """Create train/validation/test regression datasets with memory-efficient splitting."""
    print("Creating dataset splits...")
    train_indices, val_indices, test_indices = get_split_idxs(
        adata, val_split=val_split, random_state=random_state,
    )
    
    print("Densifying data matrix (this may take a moment)...")
    # Densify ONCE and share across all datasets
    X_dense = _to_dense_numpy(adata.X)
    y_data = adata.obsm[target_key]
    
    print("Creating dataset objects...")
    # Create datasets that share the same dense matrix
    train_dataset = RegressorDataset(X_dense, y_data, train_indices, device)
    val_dataset = RegressorDataset(X_dense, y_data, val_indices, device)
    test_dataset = RegressorDataset(X_dense, y_data, test_indices, device)
    
    # Store adata reference for compatibility
    train_dataset.adata = adata
    val_dataset.adata = adata
    test_dataset.adata = adata
    
    print(f"Datasets created - Train: {len(train_dataset):,}, Val: {len(val_dataset):,}, Test: {len(test_dataset):,}")
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
    