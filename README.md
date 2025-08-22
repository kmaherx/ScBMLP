# ScBMLP: Bilinear Multi-Layer Perceptrons for Single-Cell Analysis

Interpretable gene regulatory patterns through bilinear neural networks and graph Laplacian eigendecomposition.

## Concept

Captures **gene-gene interactions** rather than correlation-based co-expression patterns.

- **Bilinear Interactions**: `(Wx) ⊙ (Vx) → B ∈ ℝ^(g×g)` - pairwise gene interactions
- **Graph-Based Decomposition**: Laplacian eigenvectors create natural frequency scales
- **Multi-scale Organization**: From global cellular patterns to fine-grained markers

## Installation

```bash
pip install git+https://github.com/kmaherx/ScBMLP.git
```

## Quick Start

```python
from scripts.datasets import myeloid_freqs
from scripts.bmlp import ScBMLPRegressor, Config
import torch
import einops

# load data with frequency decomposition
adata, train_dataset, val_dataset, test_dataset = myeloid_freqs()

# configure and train model
cfg = Config(d_input=adata.n_vars, d_hidden=128, d_output=5)
model = ScBMLPRegressor(cfg)
model.fit(train_dataset, val_dataset)

# extract gene modules
b = einops.einsum(model.w_p, model.w_l, model.w_r, "out hid, hid in1, hid in2 -> out in1 in2")
b = 0.5 * (b + b.mT)

for freq_idx in range(5):
    vals, vecs = torch.linalg.eigh(b[freq_idx])
    vals, vecs = vals.flip([0]), vecs.flip([1])
    top_genes = adata.var_names[vecs[:, 0].topk(20).indices]
    print(f"Freq {freq_idx}: {top_genes.tolist()}")
```

## Key Benefits

1. **Non-linear & Interpretable**: Captures complex gene regulatory logic while maintaining interpretability
2. **Multi-scale Analysis**: Natural decomposition of cellular organization across multiple scales
3. **Beyond Correlations**: Discovers gene interactions that correlation methods miss
4. **Biologically Grounded**: Frequency decomposition aligns with cellular organization principles

## Validation Results

Our approach shows significant advantages over traditional correlation-based methods:

- **Low Jaccard Similarity (0.118)**: Demonstrates that bilinear interactions capture fundamentally different biology than simple correlations
- **GO Term Enrichment**: Gene modules show strong enrichment for relevant biological processes at each frequency
- **Multi-scale Organization**: Clear progression from global cellular patterns to specific cell type markers

## Installation

### Prerequisites

- Python ≥3.9
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
