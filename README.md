# Welcome to tensorSB repository!

This repository provides a CUDA-accelerated tensor network library with support for DMRG. It is built on top of ```PyTorch, NumPy, CuPy,``` and the [NVIDIA cuQuantum SDK].

## Freeze environment
To freeze your current environment:
```
conda list --export > conda_requirements.txt
conda env export > environment.yml
```

## Installation

Within your conda environment, run:
```
cd tensorSB
pip install -e .

```
Check installation.txt for required packages.
## Usage

API examples can be found in the `examples/` directory.


