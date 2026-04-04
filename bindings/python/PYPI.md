# forestfire

`forestfire` is a Python package for tree-based learning backed by a Rust core.

Current capabilities:

- decision trees
- random forests
- gradient boosting
- classification and regression
- optimized inference runtimes
- model introspection and export

Example:

```python
import numpy as np

from forestfire import train

X = np.array([[0.0], [0.0], [1.0], [1.0]])
y = np.array([0.0, 0.0, 1.0, 1.0])

model = train(X, y, task="classification", tree_type="cart")
print(model.predict(X))
print(model.predict_proba(X))
```

Common missing-value markers such as `None`, `np.nan`, pandas/NumPy `NaN`, and
`polars` nulls are handled automatically during training and prediction.

The source repository, documentation, and issue tracker live at:

- https://github.com/wsperat/forest-fire
