# Causal discovery

Causal discovery is the process of inferring causal relationships between variables from observational data. This repository aims to provide a collection of causal discovery algorithms implemented in Python.

## Development setup

This repository uses [Poetry](https://python-poetry.org/) as a dependency manager. To install the dependencies, run:

```zsh
$ poetry install
```

## Usage

Pull this repository and run the following command:

```zsh
$ poetry build
```

Then, install the package:

```zsh
$ pip install dist/causal-discovery-0.1.0.tar.gz
```

example usage:

```python
from causal_discovery.algos.notears import NoTears

# load dataset
dataset = ...  

# initialize model
model = NoTears(
    rho=1, 
    alpha=0.1, 
    l1_reg=0, 
    lr=1e-2
)

# learn the graph
_ = model.learn(dataset.X)

# adjacency matrix
print(model.W)
```

## Algorithms

| Algorithm | Reference |
|-----------|-----------|
| **NOTEARS** | [DAGs with NO TEARS: Continuous Optimization for Structure Learning, 2019](https://arxiv.org/abs/1803.01422) |

## Results

This is the example of the results of the algorithm.

![Results](./images/notears_res.png)
