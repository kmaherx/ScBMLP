# ScBMLP: Bilinear Multi-Layer Perceptrons for Single-Cell Analysis

A novel machine learning approach for discovering interpretable gene regulatory patterns in single-cell RNA-seq data through bilinear neural networks and graph Laplacian eigendecomposition.

## 🧬 Concept

Traditional single-cell analysis methods rely on correlation-based approaches to identify gene programs and cell states. This project introduces a fundamentally different approach that captures **gene-gene interactions** rather than simple co-expression patterns.

### Core Innovation

- **Bilinear Interactions**: Instead of linear transformations, we use bilinear operations that model pairwise gene interactions: `(Wx) ⊙ (Vx) → B ∈ ℝ^(g×g)`
- **Graph-Based Decomposition**: We decompose the cell-cell similarity graph using Laplacian eigenvectors, creating natural frequency scales that correspond to cellular organization
- **Interpretable Weights**: The learned bilinear matrices encode gene regulatory modules at different transcriptional scales

### Biological Intuition

Each output dimension represents a different scale of cellular organization:
- **Low frequencies (0-1)**: Global cellular patterns and major cell fate decisions
- **Middle frequencies (2-3)**: Lineage-specific programs and differentiation pathways  
- **High frequencies (4+)**: Fine-grained cell type markers and specialized functions

The interaction matrices `B[freq]` represent gene modules that capture regulatory relationships at each scale.

## 🚀 Key Benefits

1. **Non-linear & Interpretable**: Captures complex gene regulatory logic while maintaining interpretability
2. **Multi-scale Analysis**: Natural decomposition of cellular organization across multiple scales
3. **Beyond Correlations**: Discovers gene interactions that correlation methods miss
4. **Biologically Grounded**: Frequency decomposition aligns with cellular organization principles

## 📊 Validation Results

Our approach shows significant advantages over traditional correlation-based methods:

- **Low Jaccard Similarity (0.118)**: Demonstrates that bilinear interactions capture fundamentally different biology than simple correlations
- **GO Term Enrichment**: Gene modules show strong enrichment for relevant biological processes at each frequency
- **Multi-scale Organization**: Clear progression from global cellular patterns to specific cell type markers

## 🛠 Installation

### Prerequisites

- Python 3.9
- Recommended: Create a virtual environment first

### Quick Install (Recommended)

Install directly from GitHub using pip:

```bash
pip install git+https://github.com/kmaherx/ScBMLP.git
```

### Virtual Environment Setup (Recommended)

For a clean installation, create a virtual environment first:

```bash
python -m venv scbmlp-env
source scbmlp-env/bin/activate  # on Windows: scbmlp-env\Scripts\activate
pip install git+https://github.com/kmaherx/ScBMLP.git
```

### Development Installation

For development or if you want to modify the code:

```bash
git clone https://github.com/kmaherx/ScBMLP.git
cd ScBMLP
pip install -e ".[dev]"
```

### Using uv (Alternative Package Manager)

If you prefer using `uv` for faster package management:

```bash
uv pip install git+https://github.com/kmaherx/ScBMLP.git
```

## 🧪 Quick Start

```python
import scanpy as sc
import numpy as np
from scripts.datasets import myeloid_dev_freq
from scripts.bmlp import ScBMLPRegressor, Config

# Load data with frequency decomposition
adata, train_dataset, val_dataset, test_dataset = myeloid_dev_freq(
    device="cpu", 
    k_neighbors=10,
    n_freq_comps=5
)

# Configure and train the model
cfg = Config(
    d_input=adata.n_vars,
    d_hidden=128,
    d_output=5,  # number of frequency components
    n_epochs=500,
    lr=1e-4
)

model = ScBMLPRegressor(cfg, loss_fn="l1")
train_losses, val_losses = model.fit(train_dataset, val_dataset)

# Extract interpretable gene modules
import einops
b = einops.einsum(model.w_p, model.w_l, model.w_r, "out hid, hid in1, hid in2 -> out in1 in2")
b = 0.5 * (b + b.mT)  # symmetrize

# Analyze gene modules at each frequency
for freq_idx in range(5):
    vals, vecs = torch.linalg.eigh(b[freq_idx])
    vals, vecs = vals.flip([0]), vecs.flip([1])
    
    # Get top genes for this frequency
    top_genes = adata.var_names[vecs[:, 0].topk(20).indices]
    print(f"Frequency {freq_idx} genes: {top_genes.tolist()}")
```

## 📖 Documentation

### Core Components

- **`scripts/bmlp.py`**: Bilinear MLP implementations with abstract base classes
- **`scripts/datasets.py`**: Data loading utilities for single-cell datasets  
- **`scripts/utils.py`**: Graph construction and frequency calculation utilities
- **`notebooks/myeloid_dev.ipynb`**: Complete analysis pipeline with biological validation

### Key Classes

- `BaseBMLP`: Abstract base class for bilinear MLPs
- `ScBMLPRegressor`: Regression variant for frequency prediction
- `ScBMLPClassifier`: Classification variant for cell type prediction
- `Config`: Configuration dataclass for hyperparameters

## 🔬 Research Applications

This method has been validated on:

- **Myeloid Development**: Paul et al. 2015 dataset showing clear developmental patterns
- **Gene Regulatory Networks**: Discovery of biologically coherent gene modules
- **Cell State Analysis**: Capturing cellular organization and transitions

### Potential Extensions

1. **Trilinear MLPs**: `(Wx) ⊙ (Vx) ⊙ (Ux)` for ternary gene interactions
2. **Disease Applications**: Finding disrupted regulatory modules in pathological states

## 📈 Performance

- **Scalability**: Efficient up to ~20,000 cells (limited by eigendecomposition)
- **Interpretability**: Direct biological interpretation of learned weights
- **Validation**: GO term enrichment confirms biological coherence

## 🤝 Contributing

Development setup:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
```
