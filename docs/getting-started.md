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
- `task rust-verify`
- `task docs-serve`
- `task docs-build`

## How to think about the API

The intended user flow is:

1. give the library a feature matrix and target
2. let `Table` decide how to represent the data
3. call the unified `train(...)` entrypoint
4. use `predict(...)` on raw inference data
5. optionally call `optimize_inference(...)` when scoring is performance-critical
6. serialize the result when you need a portable artifact

That is why the API is organized around `Table`, `train`, `predict`, `optimize_inference`, and `serialize`, rather than around many learner-specific classes.
