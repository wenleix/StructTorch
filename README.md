# Axolotls

Axolotls implements [Arrow Memory Layout](https://arrow.apache.org/docs/format/Columnar.html#physical-memory-layout) on top of PyTorch, with lightweight DataFrame-flavored API. Such structures are often useful to perform pre-processing after data is in Tensor format.

## Installation

You will need Python 3.7 or later. Also, we highly recommend installing an [Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) environment.

First, set up an environment. If you are using conda, create a conda environment:
```
conda create --name axolotls python=3.7
conda activate axolotls
```

You can then install Axolotos from source:
```
python setup.py install
```

For development, use the following command to install instead:
```
python setup.py develop
```
