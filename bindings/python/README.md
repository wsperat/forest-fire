# forestfire 🌲🔥 (python)

Python bindings for the unified `forestfire.train(...)` interface.

## Current supported surface

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

## Why the API looks like this

### `algorithm`

Only `dt` exists today, but the parameter is already present so the public API does not need to break when additional learner families are added later.

### `task`

`task` is explicit because classification and regression should not be inferred from the target array heuristically. That keeps the train path predictable and makes unsupported combinations fail early.

### `tree_type`

`tree_type` selects the structural family of tree you want, not just a minor variation:
- `target_mean` is the simplest regression baseline
- `id3` and `c45` are explicit information-based classifiers
- `cart` is the standard binary-tree family
- `oblivious` uses the same split across a depth, which gives a more regular tree shape

### `criterion`

`criterion` is exposed because it changes the model’s bias:
- `gini` and `entropy` are the classification choices
- `mean` and `median` are the regression choices
- `auto` resolves to the family-appropriate default

Current `auto` behavior:
- `id3`, `c45` -> `entropy`
- classification `cart`, `oblivious` -> `gini`
- regression models -> `mean`

### `canaries`

The library uses automatic growth stopping rather than pruning. To do that, `DenseTable` builds shuffled canary copies of the binned features. If a learner prefers a canary feature, it has found a noise-like split and stops growing there. For oblivious trees, selecting a canary stops the whole remaining growth process.

### `physical_cores`

This controls CPU usage using physical cores rather than logical threads, because the training work is mostly memory-sensitive split scoring. Using physical cores tends to be a more honest and predictable knob for tree training.

## Data and training rationale

Training converts the NumPy input into an internal Arrow-backed `DenseTable`.

That design exists for three reasons:
- columnar scans are a better fit than row-oriented storage for repeated feature scoring
- binary `0/1` columns can be stored compactly as booleans and split faster
- pre-binning numeric columns into 512 bins makes repeated split evaluation substantially cheaper

The bin count is fixed today because the project is optimizing for a stable, simple training core first. `512` is meant as a practical compromise between split resolution and memory/runtime cost.

Training parallelism is feature-parallel by tree type:
- `id3`, `c45`, and `cart` score features in parallel at each node
- `oblivious` scores features in parallel at each tree level
- `target_mean` is effectively sequential because there is too little work to parallelize meaningfully

## Build & install (dev)
```bash
# from repo root, ensure maturin is installed: pip install maturin
cd bindings/python
maturin develop  # builds and installs into your current venv
