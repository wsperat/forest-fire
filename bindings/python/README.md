# forestfire (Python)

Python bindings for ForestFire.

The Python package exposes:

- `Table`
- `train(...)`
- `Model`
- `OptimizedModel`

## Install

```bash
pip install forestfire-ml
```

Import:

```python
import forestfire
```

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

## Documentation

The detailed Python documentation now lives in the docs site:

- published: [https://wsperat.github.io/forest-fire/](https://wsperat.github.io/forest-fire/)
- source page: [../../docs/python-api.md](../../docs/python-api.md)

That includes:

- training parameters
- supported input types
- missing-value handling semantics
- optimized inference
- selective missing checks via `optimize_inference(..., missing_features=...)`
- used-feature projection in optimized models
- serialization
- tree introspection
- dataframe export
