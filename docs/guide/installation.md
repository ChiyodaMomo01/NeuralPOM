# Installation

## Python environment

**Python 3.10 or newer** is required. Install a CUDA-capable PyTorch distribution for GPU training, then install NeuralPOM from the repository root:

```bash
git clone https://github.com/ChiyodaMomo01/NeuralPOM.git
cd NeuralPOM
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

The package declares PyTorch, NumPy, Matplotlib, W&B and netCDF4 dependencies. The `test` extra adds pytest and timm; the `ai` extra adds timm for the direct neural models. The dependency versions and console commands are defined in [pyproject.toml](https://github.com/ChiyodaMomo01/NeuralPOM/blob/main/pyproject.toml).

## Training environment

The example configurations use **eight DDP ranks**. Set the following limits before launching the commands in [Quick start](./quickstart.md):

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
```

The Explicit templates use eager physics. The semi-implicit templates select the registered compiled vertical operator and Triton sea-surface backend. The reference ensembles, model weights and matching training assets are supplied separately from the code checkout; see [Reproduction](../reproduction.md).

## Installation checks

```bash
python -c "import neuralpom; from neuralpom.core_factory import CORE_REGISTRY; print(sorted(CORE_REGISTRY))"
python -m pytest tests/test_v1_sampling.py
```

The registered core names are `explicit` and `implicit_p0_fixed`.
