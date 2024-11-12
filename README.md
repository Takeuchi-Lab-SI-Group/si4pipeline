# si4pipeline

[![PyPI - Version](https://img.shields.io/pypi/v/si4pipeline)](https://pypi.org/project/si4pipeline/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/si4pipeline)](https://pypi.org/project/si4pipeline/)
[![PyPI - License](https://img.shields.io/pypi/l/si4pipeline)](https://opensource.org/license/MIT)

This package provides the statistical test for any feature selection pipeline by selective inference.
The tequnical details are described in the paper "[Statistical Test for Feature Selection Pipelines by Selective Inference](https://arxiv.org/abs/2406.18902)".

## Installation & Requirements
This package has the following dependencies:
- Python (version 3.10 or higher, we use 3.12.5)
    - numpy (version 1.26.4 or higher but lower than 2.0.0, we use 1.26.4)
    - scikit-learn (version 1.5.1 or higher, we use 1.5.1)
    - sicore (version 2.3.0 or higher, we use 2.3.0)
    - tqdm (version 4.66.5 or higher, we use 4.66.5)

To install this package, please run the following commands (dependencies will be installed automatically):
```bash
$ pip install si4pipeline
```

## Usage
The implementation we developed can be interactively executed using the provided `demonstration.ipynb` [file](https://github.com/Takeuchi-Lab-SI-Group/si4pipeline/blob/main/demonstration.ipynb) in our repository.
This file contains a step-by-step guide on how to use the package, how to construct a feature selection pipeline, and how to apply the proposed method to a given feature selection pipeline.
