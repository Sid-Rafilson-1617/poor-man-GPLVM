# Poor Man's GPLVM

A simplified implementation of Gaussian Process Latent Variable Models (GPLVM). This package provides an easy-to-use interface for dimensionality reduction and visualization using GPLVMs.

## Installation

You can install the package from PyPI (once published):

### Basic Installation
```bash
pip install poor-man-gplvm
```

### With JAX CPU Support
```bash
pip install poor-man-gplvm[cpu]
```

### With JAX GPU Support
```bash
pip install poor-man-gplvm[gpu]
```

Note: For GPU support, you need compatible CUDA and cuDNN installations. The package assumes CUDA 12 and cuDNN 8.9 by default. If you have different versions, please install JAX and JAXlib separately following [JAX installation instructions](https://github.com/google/jax#installation).

For more detailed installation instructions, including conda environments and troubleshooting, see [INSTALL.md](INSTALL.md).

### From Source
You can also install directly from the repository:

```bash
# Basic installation
pip install git+https://github.com/samdeoxys1/poor-man-GPLVM.git

# With CPU JAX support
pip install "git+https://github.com/samdeoxys1/poor-man-GPLVM.git#egg=poor-man-gplvm[cpu]"

# With GPU JAX support
pip install "git+https://github.com/samdeoxys1/poor-man-GPLVM.git#egg=poor-man-gplvm[gpu]"
```

### From Conda Environment
If you're using conda, you might want to set up your environment first:

```bash
# Create a new conda environment
conda create -n gplvm python=3.9
conda activate gplvm

# For CPU installation
pip install poor-man-gplvm[cpu]

# For GPU installation, first install CUDA and cuDNN through conda
conda install -c nvidia cuda=12.0 cudnn=8.9
pip install poor-man-gplvm[gpu]
```

## Usage

Here's a quick example of how to use the package:

```python
import numpy as np
from poor_man_gplvm.core import GPLVM

# Generate some random high-dimensional data
Y = np.random.randn(100, 10)

# Initialize and fit the GPLVM model
model = GPLVM(latent_dim=2)
model.fit(Y)

# Access the latent variables
X = model.X

# Transform new data
Y_new = np.random.randn(5, 10)
X_new = model.transform(Y_new)
```

## Features

- Simple and intuitive API
- Efficient implementation of GPLVM
- Customizable kernel functions
- PCA-based initialization
- Visualization tools
- Optional JAX acceleration (CPU or GPU)

## Development

### Setting up the development environment

```bash
git clone https://github.com/samdeoxys1/poor-man-GPLVM.git
cd poor-man-GPLVM
pip install -e ".[dev]"

# For JAX CPU development
pip install -e ".[dev,cpu]"

# For JAX GPU development
pip install -e ".[dev,gpu]"
```

### Running tests

```bash
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```
@software{poor_man_gplvm,
  author = {Zheyang Sam Zheng},
  title = {Poor Man's GPLVM: A simplified implementation of Gaussian Process Latent Variable Models},
  year = {2024},
  url = {https://github.com/samdeoxys1/poor-man-GPLVM}
}
```
