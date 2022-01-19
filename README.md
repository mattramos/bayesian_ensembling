# Codebase for Bayesian Ensembling Project

[![codecov](https://codecov.io/gh/thomaspinder/bayesian_ensembling/branch/master/graph/badge.svg?token=0UOAPN98TV)](https://codecov.io/gh/thomaspinder/bayesian_ensembling)

General structure is as follows:

- Source code is found in `ensembles/`
- Tutorial notebooks for using the code can be found in `nbs/`
- Experimental files in `experiments/`
- Unit test files in `tests/`

## Getting started

Use the supplied `environment.yml` file to set up a local environment with the
correct dependencies. We recommend using conda and the following commands

```bash
conda env create -f environment.yml
conda activate bayesian_ensembles
```

We make use of code not available on PyPi. To install this, please navigate
towards each subdirectory within `local_installs` and run `python setup.py develop` or `python setup.py install` depending on if you are wishing to extend
and develop this code yourself, or simply install it for personal use.

Finally, the code within this repository should be installed through `python setup.py develop`.

### Validation

One can validate that their environment has been correctly set up by running
`make tests` and ensuring that no errors are reported.

## Code principles

### Formatting

Before commiting code, run `make clean` from the repository's root to black
format all code and arange imports.

### Notebooks

Notebooks should be commited in JupyText's `.py` format to allow for better
versioning. Notebooks can be built locally by running `make notebooks`

### Tests

Tests are helpful but by no means a requirement. PyTest is used for unit tests.
