# Axolotls: a data processing library for PyTorch

Axolotls is still under early development. It is a lightweight Python DataFrame library for data preprocessing in [PyTorch](https://github.com/pytorch/pytorch) models, with two high-level features:
* DataFrame library (like Pandas) bulit on PyTorch Tensor. This provides strong GPU accleration and PyTorch ecosystem integration.
* Columnar memory layout based on [Apache Arrow](https://arrow.apache.org/docs/format/Columnar.html#physical-memory-layout). This provides strong variable-width and nested data support (such as string, list, map) and Arrow ecosystem integration.

Usually, Axolotls is used to author data preprocessing module for PyTorch models that provides maximum flexibility and speed.

## Installation

You will need Python 3.7 or later. Also, we highly recommend installing an [Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) environment.

First, set up an environment. If you are using conda, create a conda environment:
```
conda create --name axolotls python=3.8
conda activate axolotls
```

Install Axolotos from source:
```
pip install --user "git+https://github.com/wenleix/axolotls"
```
