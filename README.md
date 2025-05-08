# EOTorch


## Installation

### Basic Installation (CPU-only)
To install the base package (with core functionality, excluding extras):

```bash
pip install git+https://github.com/DHI/eotorch.git
```

## Optional Dependencies

EOTorch supports several optional features. You can install them via [PEP 508](https://peps.python.org/pep-0508/) extras.

### Available extras

| Extra       | Description                                 |
|-------------|---------------------------------------------|
| `dev`       | Developer tools (formatting, testing, docs) |
| `test`      | Testing tools (pytest, coverage, mypy)      |
| `notebooks` | Jupyter notebook support                    |
| `cuda126`   | GPU support for Windows with CUDA 12.6      |

### Example: Install with optional dependencies for Jupyter notebook and Windows GPU Support
```bash
pip install "git+https://github.com/DHI/eotorch.git[cuda126,notebooks]"
```

## ⚠ Windows GPU Installation Notes

If you're using **Windows with an NVIDIA GPU**, follow these guidelines:

- Your **GPU driver** must support **CUDA 12.6**. You can check your driver compatibility by running:

  ```bash
  nvidia-smi
  ```

  Look for the `Driver Version` and `CUDA Version`. As long as the driver supports CUDA **12.6 or newer**, the `cuda126` optional dependency will work.

- The `cuda126` optional dependency installs:
  - `torch>=2.7.0`
  - `torchvision>=0.22.0`
  using the official [PyTorch CUDA 12.6 wheels](https://download.pytorch.org/whl/cu126).
---

## Development
To set up a local development environment for a new project using UV:

```bash
uv init
uv add "git+https://github.com/DHI/eotorch.git[<add_extras_here>]"
```

---

## Examples

You can find usage examples in the [notebooks](notebooks/) directory.

Example: [Segmentation with TorchGeo](notebooks/segmentation.ipynb)

---
