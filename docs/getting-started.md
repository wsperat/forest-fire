# Getting Started

## Python

```python
import numpy as np

from forestfire import train

X = np.array([[0.0], [0.0], [1.0], [1.0]])
y = np.array([0.0, 0.0, 1.0, 1.0])

model = train(X, y, task="classification", tree_type="cart")
print(model.predict(X))
print(model.predict_proba(X))
```

Install:

```bash
pip install forestfire-ml
```

Import:

```python
import forestfire
```

## Rust

```rust
use forestfire_core::{train, Criterion, Task, TrainAlgorithm, TrainConfig, TreeType};
use forestfire_data::Table;

let x = vec![vec![0.0], vec![0.0], vec![1.0], vec![1.0]];
let y = vec![0.0, 0.0, 1.0, 1.0];
let table = Table::new(x.clone(), y)?;

let model = train(
    &table,
    TrainConfig {
        algorithm: TrainAlgorithm::Dt,
        task: Task::Classification,
        tree_type: TreeType::Cart,
        criterion: Criterion::Gini,
        ..TrainConfig::default()
    },
)?;
```

## Local development

```bash
task setup-local-env
task python-ext-develop
```

Useful tasks:

- `task test`
- `task verify`
- `task rust-verify`
- `task docs-serve`
- `task docs-build`

Verification note:

- `task verify` includes the Python extension build path, not just Rust checks
- if your environment is offline and the needed Python wheels are not already cached, `task verify` can fail during `task python-ext-develop` because `maturin develop` may need to install dependencies such as `numpy`

## How to think about the API

The intended user flow is:

1. give the library a feature matrix and target
2. let `Table` decide how to represent the data
3. call the unified `train(...)` entrypoint
4. use `predict(...)` on raw inference data rather than rebuilding a training table
5. optionally inspect `used_feature_indices` to see what the trained model actually depends on
6. optionally call `optimize_inference(...)` when scoring is performance-critical
7. serialize the semantic model or snapshot the optimized runtime depending on deployment needs

That is why the API is organized around `Table`, `train`, `predict`, `optimize_inference`, and `serialize`, rather than around many learner-specific classes.

The most important conceptual distinction is:

- `Table` is primarily for training-time normalization and preprocessing
- inference should usually consume raw user-facing inputs directly
- optimized inference derives its own projected runtime representation from the trained model
