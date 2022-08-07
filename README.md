# Axolotls

Axolotls is still under early development. It is a Python package for data preprocessing in PyTorch models. It provides two high-level features:
* Lightweight DataFrame (like Pandas) with [Arrow Memory Layout](https://arrow.apache.org/docs/format/Columnar.html#physical-memory-layout) implemented on top of PyTorch Tensors. 
* Support variable-width and nested data types such as string, list, map.

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
