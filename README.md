# ForestFire
![ForestFire](docs/forest-fire.jpg)
## Fast tree-based learning in Rust

A tree-learning library with a Rust core and a Python API.
The current implementation focuses on a unified `train` interface, Arrow-backed dense tables, and a small set of tree learners.

# Why this library?

- Performance-first: Rust + Rayon for parallel training/prediction; cache-friendly data layouts.
- Arrow-backed storage: `DenseTable` stores feature columns in Arrow arrays and pre-bins numeric features into 512 bins.
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
- Arrow-backed `DenseTable` with binary-column compaction and 512-bin numeric feature binning
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
- NumPy input support in Python

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

from forestfire import train

X = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
y = np.array([0.0, 0.0, 0.0, 1.0])

clf = train(
    X,
    y,
    algorithm="dt",
    task="classification",
    tree_type="cart",
    criterion="gini",
    canaries=2,
    physical_cores=4,
)
pred = clf.predict(X)            # -> NumPy array [n_samples]
```

# Train Interface

`train(X, y, algorithm=..., task=..., tree_type=..., criterion=..., canaries=..., physical_cores=...)`

NumPy arrays are accepted directly for `train(X, y, ...)`.

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

Training first materializes an Arrow-backed `DenseTable`.

Why `DenseTable` exists:
- learners need a validated, columnar, reusable training view
- the Python and Rust APIs should feed the same internal representation
- repeated split scoring benefits from a layout designed for scans rather than row-by-row iteration

Why Arrow-backed columns:
- columnar arrays are cache-friendly for feature-wise scoring
- Arrow gives compact primitive storage and a clear path to future interop work

Why binary columns are stored as booleans:
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
use forestfire_data::DenseTable;

fn main() -> Result<()> {
    let x = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let y = vec![0.0, 0.0, 0.0, 1.0];
    let table = DenseTable::new(x, y)?;
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
- Arrow everywhere: the core stores features in Arrow arrays for columnar, cache-friendly scans.
- `DenseTable` holds feature columns and target data separately so training and inference logic can stay focused on model behavior.
- Tree learners consume the pre-binned representation rather than re-sorting features during fitting.
- Binary features receive a dedicated compact representation because they are common and structurally simpler to split on.
- Canary columns are synthetic shuffled copies of the binned features and act as a built-in stopping signal.
- Automatic growth stopping is preferred over pruning because it keeps the stopping rule inside the split search itself.
- The current Python API is NumPy-first because that gives the smallest dependency surface while the core learning design stabilizes.

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
No. The current Python binding accepts NumPy arrays directly.

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
