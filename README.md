# si4automl
This package provides the statistical test for any data analysis pipeline by selective inference.
The tequnical details are described in the paper "[Statistical Test for Data Analysis Pipeline by Selective Inference](https://arxiv.org/abs/2406.18902)".

## Installation & Requirements
This package has the following dependencies:
- Python (version 3.10 or higher, we use 3.12.5)
    - numpy (version 1.26.4 or higher but lower than 2.0.0, we use 1.26.4)
    - scikit-learn (version 1.5.1 or higher, we use 1.5.1)
    - sicore (version 2.0.3 or higher, we use 2.0.3)
    - tqdm (version 4.66.5 or higher, we use 4.66.5)

To install this package, please run the following commands (dependencies will be installed automatically):
```bash
$ pip install si4automl
```

## Usage

The implementation we developed can be interactively executed using the provided `demonstration.ipynb` file.
This file contains a step-by-step guide on how to use the package, how to construct a data analysis pipeline, and how to apply the proposed method to a given data analysis pipeline.
