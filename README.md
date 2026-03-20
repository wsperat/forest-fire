# ForestFire
![ForestFire](docs/forest-fire.jpg)

## Fast tree-based learning in Rust

ForestFire is a tree-learning library with a Rust core and a Python API. The current implementation is centered around three ideas:

- one unified `train(...)` interface instead of learner-specific entrypoints
- one unified `Table` abstraction that chooses the right internal layout automatically
- one explicit JSON IR for serialization, portability, and future runtime/export work

## What exists today

- Unified `train` API in Rust and Python
- Automatic table selection between `DenseTable` and `SparseTable`
- Classification trees: `id3`, `c45`, `cart`, `oblivious`
- Regression trees: `target_mean`, `cart`, `oblivious`
- Criterion selection via `gini`, `entropy`, `mean`, `median`, or `auto`
- Canary-based automatic growth stopping
- Physical-core-aware parallel training
- JSON serialization, deserialization, and formal JSON Schema for the IR
- Python ingestion from NumPy, pandas, polars, pyarrow, and SciPy

## Quickstart

### Python

```python
import numpy as np

from forestfire import Table, train

X = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
y = np.array([0.0, 0.0, 0.0, 1.0])

table = Table(X, y, canaries=2)

model = train(
    table,
    algorithm="dt",
    task="classification",
    tree_type="cart",
    criterion="gini",
    physical_cores=4,
)

preds = model.predict(table)
serialized = model.serialize(pretty=True)
restored = model.deserialize(serialized)
ir_json = model.to_ir_json(pretty=True)
```

### Rust

```rust
use anyhow::Result;
use forestfire_core::{train, Criterion, Task, TrainAlgorithm, TrainConfig, TreeType};
use forestfire_data::Table;

fn main() -> Result<()> {
    let x = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let y = vec![0.0, 0.0, 0.0, 1.0];

    let table = Table::new(x, y)?;
    let model = train(
        &table,
        TrainConfig {
            algorithm: TrainAlgorithm::Dt,
            task: Task::Classification,
            tree_type: TreeType::Cart,
            criterion: Criterion::Gini,
            physical_cores: Some(4),
        },
    )?;

    let preds = model.predict_table(&table);
    let serialized = model.serialize_pretty()?;
    let restored = forestfire_core::Model::deserialize(&serialized)?;

    assert_eq!(preds, restored.predict_table(&table));
    Ok(())
}
```

## Installation

### Python development install

Prerequisites:

- Rust stable toolchain
- Python `>= 3.12`

From the repo root:

```bash
task setup-local-env
task python-ext-develop
```

If you only want the wheel:

```bash
task python-ext-wheel
```

### Rust usage

Use the core and data crates directly from the workspace today. The library is still early-stage, so treat the repo state as the source of truth for the public surface.

## How to think about the API

The intended user flow is:

1. give the library a feature matrix and target
2. let `Table` decide how to represent the data
3. call the unified `train(...)` entrypoint
4. use `predict(...)`
5. serialize the result to the JSON IR when you need a portable artifact

That is why the API is organized around `Table`, `train`, `predict`, and `serialize`, rather than around many learner-specific classes.

## Training interface

Python:

```python
train(
    X_or_table,
    y=None,
    algorithm="dt",
    task="regression",
    tree_type="target_mean",
    criterion="auto",
    canaries=2,
    physical_cores=None,
)
```

Rust:

```rust
train(&table, TrainConfig { ... })
```

### Parameters

#### `algorithm`

Current value:

- `dt`

Why it exists:

- It keeps the top-level API stable as the library grows beyond a single learner family.
- It separates the learner family from the exact tree structure.

Why it is a string in Python but an enum in Rust:

- Python gets a familiar LightGBM-style interface.
- Rust keeps typed validation and exhaustive dispatch.

#### `task`

Current values:

- `regression`
- `classification`

Why it exists:

- ForestFire does not guess the task from `y`.
- Task choice changes leaf semantics, scoring, defaults, and supported tree types.

#### `tree_type`

Current support:

- regression: `target_mean`, `cart`, `oblivious`
- classification: `id3`, `c45`, `cart`, `oblivious`

Why it exists:

- Tree structure is one of the most meaningful modeling decisions.
- Users should choose the family explicitly instead of inferring it indirectly from many secondary knobs.

What each one means:

- `target_mean`: simplest regression baseline
- `id3`: entropy-driven multiway-style classifier
- `c45`: practical extension of ID3
- `cart`: conventional binary tree
- `oblivious`: symmetric tree with one shared split per depth

#### `criterion`

Current values:

- classification: `gini`, `entropy`, `auto`
- regression: `mean`, `median`, `auto`

Why it exists:

- Criterion changes model bias, not just speed.
- It should be explicit because it materially changes the splits and the leaf values.

Current `auto` resolution:

- `id3`, `c45` classification -> `entropy`
- `cart`, `oblivious` classification -> `gini`
- regression models -> `mean`

#### `canaries`

Default:

- `2`

Why it exists:

- ForestFire prefers automatic growth stopping over post-hoc pruning.
- Canary variables are shuffled copies of the already-binned features.
- If a canary wins a split, the learner has found a noise-like signal and stops growing there.

Current stopping behavior:

- standard trees: growth stops at that node
- oblivious trees: growth stops for the remaining tree

#### `physical_cores`

Default:

- all detected physical cores

Why it exists:

- Split scoring is CPU-bound and memory-sensitive.
- Physical cores are a better resource knob than logical threads for this workload.

Behavior:

- `None` uses all detected physical cores
- oversized values are capped
- `0` is rejected

## Table system

Training always goes through a `Table`. That layer is responsible for validation, preprocessing, and choosing the backing memory layout.

### `Table`

`Table` is the public wrapper. It inspects the feature matrix and chooses:

- `DenseTable` when any feature is non-binary
- `SparseTable` when the full matrix is binary and sparse-friendly

Why it exists:

- users should not have to decide storage layout manually
- the learners should see one shared interface
- the Python and Rust paths should converge on the same training representation

### `DenseTable`

`DenseTable` is used for mixed numeric/binary tabular data.

Key design choices:

- Arrow-backed column storage for scan-friendly feature access
- boolean storage for binary `0/1` columns to reduce memory use
- numeric rank-binning into `512` bins to make repeated split scoring cheaper
- canaries built at the table layer so every learner sees the same stopping reference

Why the numeric binning is rank-based:

- tree splits mostly care about order, not absolute scale
- rank-based bins are stable under monotonic rescaling
- it gives a compact, bounded representation for split search

### `SparseTable`

`SparseTable` is binary-only and optimized for truly sparse feature matrices.

When it is used:

- every feature is binary
- sparse storage is preferable to a dense boolean matrix
- SciPy sparse matrices map naturally onto it

How it works internally:

- each feature stores the row indices where the value is `1`
- missing row indices imply `0`
- canaries are built by shuffling the binary occupancy pattern and rebuilding those row-index lists

Why this layout is useful:

- memory scales with positive entries instead of `n_rows * n_features`
- binary splits skip threshold search entirely
- SciPy sparse inputs can be converted without dense materialization

### How learners use `DenseTable` and `SparseTable`

The learners operate against a shared `TableAccess` interface.

That means:

- split semantics stay the same across dense and sparse storage
- dense binary columns and sparse binary columns both use the binary split path
- `SparseTable` changes storage and access cost, not the public learner API

## Serialization and IR

Trained models can be:

- serialized to a JSON string
- deserialized back into a runnable model
- exported as the explicit IR JSON

Rust:

- `model.serialize()`
- `model.serialize_pretty()`
- `Model::deserialize(...)`
- `model.to_ir()`
- `model.to_ir_json()`
- `model.to_ir_json_pretty()`
- `Model::json_schema_json_pretty()`

Python:

- `model.serialize(pretty=False)`
- `Model.deserialize(serialized)`
- `model.to_ir_json(pretty=False)`

### Why the IR exists

- It makes inference semantics explicit instead of implicit in trainer internals.
- It is the stable export layer for future runtimes and exporters.
- It records preprocessing assumptions that are necessary for exact prediction.

### What IR v1 includes

- algorithm, task, tree type, and criterion
- explicit `node_tree` and `oblivious_levels` representations
- training-time numeric bin boundaries
- leaf payloads for classification and regression
- node and leaf stats:
  - sample counts
  - impurity
  - gain
  - class counts where relevant
  - variance where relevant

### What IR v1 does not include yet

- missing-value handling
- categorical preprocessing semantics
- ensembles

### JSON Schema

The formal JSON Schema for the IR lives at:

- [crates/core/schema/forestfire-ir.schema.json](/Users/waltersperat/Desktop/Personal/forest-fire/crates/core/schema/forestfire-ir.schema.json)

It is generated from the Rust IR types and checked in as a contract artifact. The test suite verifies that the generated schema matches the checked-in file exactly.

## Support matrix

### Tasks and tree types

- `task="regression"` with `tree_type="target_mean" | "cart" | "oblivious"`
- `task="classification"` with `tree_type="id3" | "c45" | "cart" | "oblivious"`

### Python input types

- NumPy arrays
- Python sequences
- pandas
- polars
- pyarrow
- SciPy dense matrices
- SciPy sparse matrices

## Design notes

- Unified train surface: the main API should stay stable even as the learner set grows.
- String API, enum core: Python stays ergonomic; Rust stays explicit.
- Shared table abstraction: learners care about semantics, not about whether storage is dense or sparse.
- Automatic growth stopping: stopping belongs inside split search, not as a later pruning pass.
- Explicit IR: model export should not depend on internal trainer state.

## Building and verification

Common tasks:

```bash
task install
task test
task verify
```

Rust-specific:

```bash
task rust-build
task rust-test
task rust-lint
```

## Roadmap

Near-term priorities:

- validation sets and callbacks in the unified train interface
- exact regression trees and more criteria
- histogram CART training
- compiled inference backends

Longer-term priorities:

- random forests
- gradient-boosted trees
- out-of-core Arrow-backed training
- categorical split optimizations
- probability calibration
- sklearn-compatible wrappers

## FAQ

### Do I need pandas or Polars?

No. NumPy arrays are enough. The extra dataframe/table inputs are convenience paths.

### Are predictions deterministic?

Yes. Given the same inputs and parameters, binning, canary construction, training, serialization, and deserialization are deterministic.

### How large can the data be?

Currently: as large as memory allows. Out-of-core training is still future work.

## Contributing

Run formatting, linting, and tests before opening changes:

```bash
task verify
```

## License

MIT
