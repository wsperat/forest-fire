# forestfire 🌲🔥 (python)

Python bindings for the unified `forestfire.train(...)` interface.

Current supported surface:
- `algorithm="dt"`
- `task="regression" | "classification"`
- `tree_type="target_mean" | "id3" | "c45" | "cart" | "oblivious"`
- `criterion="auto" | "gini" | "entropy" | "mean" | "median"`
- `canaries=2` by default for automatic growth stopping
- `physical_cores=None | int` to control training parallelism

Example:
```python
import numpy as np
from forestfire import train

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([0.0, 0.0, 0.0, 1.0])

model = train(
    X,
    y,
    algorithm="dt",
    task="classification",
    tree_type="cart",
    criterion="gini",
    physical_cores=4,
)
preds = model.predict(X)
```

Training parallelism is feature-parallel:
- `id3`, `c45`, and `cart` score features in parallel at each node
- `oblivious` scores features in parallel at each tree level

## Build & install (dev)
```bash
# from repo root, ensure maturin is installed: pip install maturin
cd bindings/python
maturin develop  # builds and installs into your current venv
