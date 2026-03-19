# ForestFire
![ForestFire](docs/forest-fire.jpg)
## Fast tree-based learning in Rust

A tree-learning library with a Rust core and a Python API.
The current implementation focuses on a unified `train` interface, automatic dense-vs-sparse table selection, and a small set of tree learners.

# Why this library?

- Performance-first: Rust + Rayon for parallel training/prediction; cache-friendly data layouts.
- Adaptive storage: `Table` chooses between Arrow-backed `DenseTable` storage and binary `SparseTable` storage.
- Friendly Python API: a LightGBM-like `train(X, y, algorithm=..., task=..., tree_type=..., criterion=..., physical_cores=...)` entrypoint.
- Automatic growth stopping: shuffled canary variables are generated during table construction and halt growth when selected.
- Parallel training: standard trees parallelize feature scoring per node, and oblivious trees parallelize feature scoring per level.
- Extensible design: a common dispatcher for tree learners behind one training interface.

# Roadmap
Several of the intended features haven't been implemented yet, so here are the desired functionalities separated according to their implementation priority.

Priorities may shift in the future, and new ones added.

## Immediate Priorities
- Generalize the `train` interface with validation sets and callbacks
- Exact regression trees and additional criteria
- Histogram CART algorithm
- Compiled inference backend (`treelite` / LLVM-style codegen) for inference speedups

## Long term Priorities
- Random Forest (bagging, feature subsampling)
- Gradient Boosted Trees (with histogram-based split finding)
- Out-of-core training via Arrow streaming/memory-mapping
- Calibrated probabilities, monotonic constraints, categorical split optimizations
- Sklearn estimator wrappers (get_params, set_params, sklearn compat tests)

# Project layout
```
.                              — Workspace root (Rust + Python)
├── bindings                   — Language bindings
│   └── python                 — PyO3-based Python extension (PEP 517 via maturin)
│       ├── Cargo.toml         — Rust crate manifest for the Python module
│       ├── forestfire.pyi     — Python type hints/stubs for IDEs & static typing
│       ├── pyproject.toml     — Python build config (maturin backend, wheel settings)
│       ├── README.md          — PyPI-facing README for the Python package
│       └── src
│           └── lib.rs         — #[pymodule] entry; exposes Rust API to Python
├── Cargo.lock                 — Resolved Rust dependencies lockfile
├── Cargo.toml                 — Workspace manifest aggregating member crates
├── crates                     — Rust workspace members (library crates)
│   ├── core                   — Core algorithms/traits (tree training logic)
│   │   ├── Cargo.toml
│   │   └── src
│   │       ├── lib.rs         — Core crate root
│   │       └── tree           — Tree model implementations
│   ├── data                   — Arrow-backed dense table & preprocessing abstractions
│   ├── exporters              — Model export backends
│   │   ├── compiled           — (Planned) codegen/compiled inference target
│   │   └── onnx               — (Planned) ONNX-ML TreeEnsemble exporter implementation
│   └── inference              — (Planned) Runtime prediction interfaces/backends
├── docs                       — Project documentation & design notes
├── examples                   — Minimal runnable examples
│   ├── python                 — Python usage of the bindings
│   └── rust                   — Rust usage of core/inference crates
├── LICENSE                    — Project license
├── pyproject.toml             — Top-level Python tooling config (e.g., uv/maturin)
├── README.md                  — Top-level overview (this file)
├── rust-toolchain.toml        — Pinned Rust toolchain for reproducible builds
├── target                     — Cargo build artifacts (ignored in VCS)
├── Taskfile.yaml              — Task runner shortcuts (build/test/lint)
├── tests                      — Cross-language tests
│   ├── python                 — Pytests for the Python API
│   └── rust                   — Rust tests for the mean tree model
└── uv.lock                    — Lockfile for Python’s uv package manager
```

# Features

- Unified `train` interface in Rust and Python
- `Table` auto-selection between `DenseTable` and `SparseTable`
- Arrow-backed `DenseTable` with binary-column compaction and 512-bin numeric feature binning
- Binary `SparseTable` specialized for truly sparse `0/1` feature matrices
- Supported tree types behind `algorithm="dt"`:
    - `target_mean`
    - `id3`
    - `c45`
    - `cart`
    - `oblivious`
- Supported tasks:
    - `regression`
    - `classification`
- Supported criteria:
    - `gini`
    - `entropy`
    - `mean`
    - `median`
    - `auto` (resolved from task and tree type)
- User-controlled training parallelism via `physical_cores`
- Automatic canary-based growth stopping with `canaries=2` by default
- NumPy, pandas, polars, pyarrow, and SciPy input support in Python

# Installation
## Prerequisites

Rust (stable, latest) and Cargo

Python >= 3.12

Required Python deps:

- numpy>=1.26

### Option A — from source with maturin
Build & install the Python package in editable/dev mode
`task python-ext-develop`

or build a wheel
`task python-ext-wheel`

### Option B — Rust only
Use the Rust core crate directly in your Cargo project
`cargo add core --git https://github.com/wsperat/forest-fire`

# Quickstart (Python)
```python
import numpy as np

from forestfire import Table, train

X = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
y = np.array([0.0, 0.0, 0.0, 1.0])

table = Table(X, y, canaries=2)  # -> "sparse" here because every feature is binary

clf = train(
    table,
    algorithm="dt",
    task="classification",
    tree_type="cart",
    criterion="gini",
    physical_cores=4,
)
pred = clf.predict(table)        # -> NumPy array [n_samples]
```

# Train Interface

`train(X, y, algorithm=..., task=..., tree_type=..., criterion=..., canaries=..., physical_cores=...)`

NumPy arrays are accepted directly for `train(X, y, ...)`, but Python callers can also build a `Table` explicitly and pass that into both `train(...)` and `predict(...)`.

## Parameter rationale

### `algorithm`

Current value: `dt`

Why it exists:
- It keeps the top-level API stable while the library grows beyond a single learner family.
- It separates "which general training family is this?" from "which exact tree shape should it use?".

Why it is a string in Python but an enum internally:
- Python callers get a LightGBM-style interface that is easy to type and inspect.
- Rust keeps exhaustive matches and validation through enums, so unsupported combinations fail clearly.

### `task`

Current values:
- `regression`
- `classification`

Why it exists:
- It forces the library to choose loss semantics explicitly instead of guessing from `y`.
- The same tree shape can behave differently under regression and classification, so task selection belongs at the API boundary.

What it controls:
- available `tree_type` values
- criterion resolution for `criterion="auto"`
- impurity/dispersion scoring
- leaf prediction semantics

### `tree_type`

Current values:
- regression: `target_mean`, `cart`, `oblivious`
- classification: `id3`, `c45`, `cart`, `oblivious`

Why it exists:
- Different tree families make different structural promises, and those differences matter more than a single "depth" or "regularization" knob.
- Exposing the tree family directly makes behavior easier to reason about than hiding it behind many secondary parameters.

Rationale for each value:
- `target_mean`: simplest regression baseline; useful as a control model and for verifying data/metric pipelines.
- `id3`: categorical-style information-gain tree; useful as a reference for entropy-first classification behavior.
- `c45`: extension of ID3 with a more practical split-selection style; kept separate because users expect it by name.
- `cart`: standard binary tree family; strong default when you want conventional decision-tree behavior.
- `oblivious`: symmetric tree where every node at a depth uses the same split; this structure is attractive for predictable inference and level-wise parallelization.

### `criterion`

Current values:
- classification: `gini`, `entropy`, `auto`
- regression: `mean`, `median`, `auto`

Why it exists:
- Criterion is one of the few choices that genuinely changes the model’s bias, not just runtime.
- Making it explicit avoids hiding a statistically meaningful decision behind undocumented defaults.

Rationale for each option:
- `gini`: usually cheaper to evaluate and a strong default for classification CART-style trees.
- `entropy`: keeps information-gain semantics explicit for `id3` and `c45`.
- `mean`: optimizes around average error behavior and matches the usual squared-error intuition.
- `median`: more robust when targets contain heavy tails or outliers.
- `auto`: keeps the API concise while still picking a criterion that matches the chosen learner family.

Current `auto` resolution:
- `id3`, `c45` classification -> `entropy`
- `cart`, `oblivious` classification -> `gini`
- regression models -> `mean`

### `canaries`

Default: `2`

Why it exists:
- This library prefers automatic growth stopping over post-hoc pruning.
- Canary features provide a built-in null reference: if a shuffled copy of a real feature looks best, the learner has reached noise territory.

Why shuffled copies:
- They preserve the marginal distribution and bin occupancy of the original feature.
- They destroy the relationship with the target.
- That makes them a stronger stopping signal than a purely synthetic random column.

How stopping works:
- standard trees: if a node selects a canary, that node becomes a leaf
- oblivious trees: if a level selects a canary, the entire tree stops growing

How to think about the value:
- `0` disables the mechanism entirely
- larger values make the stopping test harsher
- the default `2` is intentionally conservative for small trees without making toy datasets impossible to fit

### `physical_cores`

Default: all available physical cores

Why it exists:
- Tree training is CPU-bound, and users often need predictable resource usage on shared machines.
- Physical cores are a better default control than logical threads for this workload because split scoring is memory-sensitive and can saturate before SMT helps.

Behavior:
- `None` uses all detected physical cores
- values above the machine limit are capped
- `0` is rejected

Why parallelism is chosen per tree type:
- `id3`, `c45`, `cart`: feature scoring is independent at a node, so node-local feature parallelism is the cheapest win
- `oblivious`: depth-wise shared splits make per-level feature scoring the natural strategy
- `target_mean`: there is not enough work to justify thread-pool overhead

## Data rationale

Training first materializes a `Table`, which then resolves to either `DenseTable` or `SparseTable`.

Why `Table` exists:
- learners need a validated, reusable training view
- the Python and Rust APIs should feed the same internal representation
- dense numeric data and sparse binary data want different memory layouts, but the learners should not need different public APIs

### `DenseTable`

Why `DenseTable` exists:
- mixed numeric/binary tabular data benefits from feature-wise column scans
- Arrow arrays are a natural fit for dense columnar access
- repeated split scoring benefits from a layout designed for scans rather than row-by-row iteration

Why Arrow-backed columns:
- columnar arrays are cache-friendly for feature-wise scoring
- Arrow gives compact primitive storage and a clear path to future interop work

Why binary columns are stored as booleans in `DenseTable`:
- `0/1` and `false/true` features are common in tabular data
- bit-packed boolean storage cuts memory relative to generic float storage
- learners can skip threshold search and use a direct false/true split path

Why numeric features are pre-binned into 512 bins:
- tree learners repeatedly compare feature values; exact continuous-value handling is often more expensive than it is useful
- fixed bins reduce split-search cost and bound per-feature work
- `512` is a compromise: fine-grained enough for common tabular problems, still small enough to stay cheap in memory and scanning

Why binning is rank-based:
- it is stable under monotonic rescaling of the input
- it keeps the representation focused on ordering, which is what tree splits mostly care about

### `SparseTable`

When `SparseTable` is used:
- every feature column is binary
- the input is sparse enough that storing the locations of the `1`s is preferable to storing a full dense boolean matrix
- SciPy sparse matrices map naturally onto this representation

How `SparseTable` works internally:
- each feature column stores the row indices where the value is `1`
- missing row indices implicitly mean the value is `0`
- canary columns are generated by shuffling those binary occupancies and rebuilding the row-index lists
- there is no threshold search because sparse features are always binary splits

Why that layout is useful:
- memory scales with the number of positive entries rather than `n_rows * n_features`
- binary split evaluation only needs membership checks on the stored `1` positions
- SciPy sparse inputs can be converted by reading shape plus nonzero coordinates, without first materializing a dense matrix

How the trees use `SparseTable`:
- the public learners operate through a shared `TableAccess` interface
- standard trees (`id3`, `c45`, `cart`) still score one feature at a time, but sparse binary features go directly through the binary split path
- oblivious trees still score one feature per depth, and sparse columns participate exactly like dense binary columns
- from the learner’s perspective, `SparseTable` changes storage and access costs, not the split semantics

Why canaries are attached to the table instead of generated inside each learner:
- every learner should see the same stopping reference distribution
- table-level generation keeps training logic simpler and more comparable across algorithms
- it avoids repeating the same preprocessing work for each learner implementation

## Support matrix

Current task/tree support:
- `task="regression"` with `tree_type="target_mean" | "cart" | "oblivious"`
- `task="classification"` with `tree_type="id3" | "c45" | "cart" | "oblivious"`

# Quickstart (Rust)
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
    let config = TrainConfig {
        algorithm: TrainAlgorithm::Dt,
        task: Task::Classification,
        tree_type: TreeType::Cart,
        criterion: Criterion::Gini,
        canaries: 2,
        physical_cores: Some(4),
    };
    let model = train(&table, config)?;
    let preds = model.predict_table(&table);
    println!("{preds:?}");
    Ok(())
}
```

# Design notes
- Unified train surface: users should not need a different top-level entrypoint for every learner family.
- String API, enum core: Python gets ergonomic strings; Rust keeps typed dispatch and exhaustive validation.
- Shared table abstraction: the learners operate against a common table interface instead of caring whether the backing data is dense or sparse.
- Arrow where it helps: `DenseTable` stores feature columns in Arrow arrays for columnar, cache-friendly scans.
- Sparse binary features use row-index lists in `SparseTable`, so memory scales with the number of positive entries instead of the full dense shape.
- Tree learners consume the pre-binned representation rather than re-sorting features during fitting.
- Binary features receive dedicated storage and a direct split path because they are common and structurally simpler to split on.
- Canary columns are synthetic shuffled copies of the binned features and act as a built-in stopping signal.
- Automatic growth stopping is preferred over pruning because it keeps the stopping rule inside the split search itself.
- The current Python API accepts raw arrays/dataframes/sparse matrices or explicit `Table` objects, depending on how much control the caller wants over preprocessing.

# Building & testing
## Rust
Lint & test
```bash
task rust-fmt
task rust-lint
task rust-build
task rust-test
```

## Python
Build locally for your venv
```bash
task install
task setup-local-env
task lint
task test
```

# Versioning & compatibility

Rust: stable toolchain (latest two releases supported)

Python: >=3.12 (x86_64 & aarch64 on Linux/macOS/Windows)

Binary wheels: built via maturin; source installs require Rust toolchain

# Contributing

Contributions are welcome! Please:

Open an issue describing the change/feature.

Add tests and benchmarks for performance-sensitive code.

Run formatting, linting and testing tasks before submitting.

See CONTRIBUTING.md (or open an issue if you don’t see one yet).

# Acknowledgments

- PyO3 for seamless Python extensions in Rust
- Apache Arrow for columnar speed & zero-copy interop
- NumPy / pandas / Polars communities for foundational data tooling
- ONNX for portable, fast inference formats

# FAQ

## Q: Do I need pandas or Polars?
No. The Python binding accepts NumPy arrays directly, but it can also ingest pandas, Polars, pyarrow, and SciPy inputs.

## Q: Are predictions deterministic?
Yes. Binning and canary generation are deterministic for the same inputs and parameters.

## Q: How big can my data be?
As large as your machine’s memory (for now). Out-of-core/streaming training is on the roadmap.

# MIT License

Copyright (c) 2024 wsperat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
