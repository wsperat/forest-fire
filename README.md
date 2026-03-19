# ForestFire
![ForestFire](docs/forest-fire.jpg)
## Fast tree-based learning in Rust

A tree-learning library with a Rust core and a Python API.
The current implementation focuses on a unified `train` interface, Arrow-backed dense tables, and a small set of tree learners.

# Why this library?

- Performance-first: Rust + Rayon for parallel training/prediction; cache-friendly data layouts.
- Arrow-backed storage: `DenseTable` stores feature columns in Arrow arrays and pre-bins numeric features into 512 bins.
- Friendly Python API: a LightGBM-like `train(X, y, algorithm=..., tree_type=...)` entrypoint.
- Automatic growth stopping: shuffled canary variables are generated during table construction and halt growth when selected.
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
- Arrow-backed `DenseTable` with 512-bin numeric feature binning
- Supported tree types behind `algorithm="dt"`:
    - `target_mean`
    - `id3`
    - `c45`
    - `cart`
    - `oblivious`
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

clf = train(X, y, algorithm="dt", tree_type="cart", canaries=2)
pred = clf.predict(X)            # -> NumPy array [n_samples]
```

# Data interop tips

NumPy arrays are accepted directly for `train(X, y, ...)`.

Training materializes an Arrow-backed `DenseTable`, pre-bins numerical columns into 512 rank bins, and appends shuffled canary copies of the binned columns.

If a split chooses a canary variable, growth stops automatically at that node. For oblivious trees, selecting a canary stops the whole tree-growth loop.

# Quickstart (Rust)
```rust
use anyhow::Result;
use forestfire_core::{train, TrainAlgorithm, TrainConfig, TreeType};
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
        tree_type: TreeType::Cart,
    };
    let model = train(&table, config)?;
    let preds = model.predict_table(&table);
    println!("{preds:?}");
    Ok(())
}
```

# Design notes
- Arrow everywhere: The core stores features in Arrow arrays for columnar, cache-friendly scans.
- `DenseTable` holds feature columns and target data separately.
- Tree learners consume the pre-binned representation rather than re-sorting features during fitting.
- Canary columns are synthetic shuffled copies of the binned features and act as a built-in stopping signal.
- The current Python API is NumPy-first.

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

Python: >=3.13 (x86_64 & aarch64 on Linux/macOS/Windows)

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
