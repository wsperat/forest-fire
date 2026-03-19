# forestfire 🌲🔥 (python)

Python bindings for the unified `forestfire.train(...)` interface.

Current supported surface:
- `algorithm="dt"`
- `tree_type="target_mean" | "id3" | "c45" | "cart" | "oblivious"`
- `canaries=2` by default for automatic growth stopping

Example:
```python
import numpy as np
from forestfire import train

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([0.0, 0.0, 0.0, 1.0])

model = train(X, y, algorithm="dt", tree_type="cart")
preds = model.predict(X)
```

## Build & install (dev)
```bash
# from repo root, ensure maturin is installed: pip install maturin
cd bindings/python
maturin develop  # builds and installs into your current venv
