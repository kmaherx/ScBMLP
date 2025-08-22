"""
BMLP: Bilinear Multi-Layer Perceptrons for Hierarchical Single-Cell Analysis
"""

from .bmlp import BaseBMLP, ScBMLPRegressor, ScBMLPClassifier, Config
from .datasets import CellFreqDataset, myeloid_dev_freq
from .utils import get_freqs, calculate_knn_adjacency, get_cell_types

__version__ = "0.1.0"
__all__ = [
    "BaseBMLP",
    "ScBMLPRegressor", 
    "ScBMLPClassifier",
    "Config",
    "CellFreqDataset",
    "myeloid_dev_freq",
    "get_freqs",
    "calculate_knn_adjacency",
    "get_cell_types",
]
