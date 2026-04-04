# ForestFire

![ForestFire](docs/forest-fire.jpg)

Tree-based learning in Rust with a Python API.

ForestFire is organized around:

- one unified `train(...)` interface
- one shared `Table` abstraction for preprocessing and training data
- one explicit model IR for serialization and optimized inference

Current capabilities:

- decision trees
- random forests
- gradient boosting
- classification and regression
- automatic missing-value handling during training and prediction
- optimized inference runtimes
- compiled optimized runtime artifacts
- model introspection and dataframe export

## Documentation

The full documentation lives in the docs site:

- published: [https://wsperat.github.io/forest-fire/](https://wsperat.github.io/forest-fire/)
- local serve: `task docs-serve`
- local build: `task docs-build`

Main sections:

- [Getting Started](docs/getting-started.md)
- [Python API](docs/python-api.md)
- [Rust API](docs/rust-api.md)
- [Training](docs/training.md)
- [Models And Introspection](docs/models.md)
- [Runtime And Optimization](docs/runtime.md)
- [Benchmarks](docs/benchmarks.md)
- [PyPI Release](docs/pypi-release.md)
- [crates.io Release](docs/cargo-release.md)

## Quickstart

```python
import numpy as np

from forestfire import train

X = np.array([[0.0], [0.0], [1.0], [1.0]])
y = np.array([0.0, 0.0, 1.0, 1.0])

model = train(X, y, task="classification", tree_type="cart")
print(model.predict(X))
print(model.predict_proba(X))
```

Missing values are handled automatically from common Python inputs such as
`None`, `np.nan`, pandas/NumPy `NaN`, and `polars` nulls.

Training can also choose between a fast missing-value split heuristic and a
slower optimal search through `missing_value_strategy=...`.

Install from PyPI:

```bash
pip install forestfire-ml
```

Import in Python:

```python
import forestfire
```

## Development

```bash
task setup-local-env
task python-ext-develop
task test
task verify
task rust-verify
```

Notes:

- `task verify` runs the full repository checks, including Python extension rebuilds.
- if the local environment does not already have the required Python wheels cached, the `python-ext-develop` step may need network access because `maturin develop` can trigger package installation for build/runtime dependencies such as `numpy`

Repository:

- [https://github.com/wsperat/forest-fire](https://github.com/wsperat/forest-fire)
