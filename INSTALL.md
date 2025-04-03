# Installation Guide

This document provides detailed installation instructions for the Poor Man's GPLVM package, including how to set up the package with different JAX backends (CPU or GPU).

## Prerequisites

- Python 3.7 or newer
- pip package manager

### For GPU Support
- CUDA Toolkit (recommended version: 12.x)
- cuDNN (recommended version: 8.9.x)

## Installation Methods

### 1. Installing from PyPI

#### Basic Installation
```bash
pip install poor-man-gplvm
```

#### With JAX CPU Support
```bash
pip install poor-man-gplvm[cpu]
```

#### With JAX GPU Support
```bash
pip install poor-man-gplvm[gpu]
```

### 2. Installing from Source

#### Basic Installation
```bash
git clone https://github.com/samdeoxys1/poor-man-GPLVM.git
cd poor-man-GPLVM
pip install -e .
```

#### With JAX CPU Support
```bash
git clone https://github.com/samdeoxys1/poor-man-GPLVM.git
cd poor-man-GPLVM
pip install -e ".[cpu]"
```

#### With JAX GPU Support
```bash
git clone https://github.com/samdeoxys1/poor-man-GPLVM.git
cd poor-man-GPLVM
pip install -e ".[gpu]"
```

### 3. Installation with Conda

Conda can be used to manage both Python and system-level dependencies, which is especially useful for GPU support.

#### Create and Activate Environment
```bash
conda create -n gplvm python=3.9
conda activate gplvm
```

#### For CPU Installation
```bash
pip install poor-man-gplvm[cpu]
```

#### For GPU Installation with Conda-managed CUDA
```bash
# Install CUDA and cuDNN via conda
conda install -c nvidia cuda=12.0 cudnn=8.9

# Install the package with GPU support
pip install poor-man-gplvm[gpu]
```

## Custom JAX Installation

If you need specific JAX versions or have custom CUDA requirements, you might want to install JAX separately:

```bash
# First install the package without JAX
pip install poor-man-gplvm

# Then install JAX with your specific requirements
pip install jax==0.4.26
pip install jaxlib==0.4.26  # For CPU

# OR for GPU with specific CUDA version (example):
pip install jaxlib==0.4.26+cuda12.cudnn89
```

Refer to the [official JAX installation guide](https://github.com/google/jax#installation) for more details on installing JAX with specific CUDA configurations.

## Verifying Installation

To verify the installation is working correctly:

```bash
python -c "import poor_man_gplvm; print(poor_man_gplvm.__version__)"
```

For JAX installations:

```bash
# Check JAX and GPU availability
python -c "import jax; print('JAX version:', jax.__version__); print('Available devices:', jax.devices())"
```

## Troubleshooting

### CUDA Version Mismatch
If you encounter errors related to CUDA versions, ensure your CUDA toolkit version matches the jaxlib+cuda version you've installed.

### Memory Issues with GPU
If you encounter GPU memory errors, you might need to limit JAX's GPU memory usage:

```python
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
```

### Import Errors
If you encounter import errors, ensure all dependencies are properly installed:

```bash
pip install -r requirements.txt
``` 