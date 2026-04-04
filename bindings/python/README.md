# forestfire (Python)

Python bindings for ForestFire.

The Python package exposes:

- `Table`
- `train(...)`
- `Model`
- `OptimizedModel`
- sklearn-compatible wrappers in `forestfire.tree`, `forestfire.forest`, and `forestfire.gbm`

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

Sklearn-style estimators:

```python
from forestfire.tree import CARTClassifier
from forestfire.forest import CARTRandomForestRegressor
from forestfire.gbm import ObliviousGBMRegressor

classifier = CARTClassifier(max_depth=4).fit(X, y)
```

## Documentation

The detailed Python documentation now lives in the docs site:

- published: [https://wsperat.github.io/forest-fire/](https://wsperat.github.io/forest-fire/)
- source page: [../../docs/python-api.md](../../docs/python-api.md)

That includes:

- training parameters
- `bins=...` vs `histogram_bins=...`
- sklearn-compatible estimator wrappers
- supported input types
- missing-value handling semantics
- `missing_value_strategy="heuristic" | "optimal" | {...}`
- optimized inference
- selective missing checks via `optimize_inference(..., missing_features=...)`
- used-feature projection in optimized models
- serialization
- tree introspection
- dataframe export
