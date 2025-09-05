# ForestFire — fast tree-based ML in Rust with Python bindings

A high-performance library for decision trees and tree ensembles with a Rust core and a clean Python API.
Designed for speed, correctness, and interoperability with NumPy, pandas, Polars, and Arrow.

# Why this library?

- Performance-first: Rust + Rayon for parallel training/prediction; cache-friendly data layouts.
- Zero-copy I/O: Uses Apache Arrow internally so Python <-> Rust data can flow without copies.
- Friendly Python API: native, lightgbm-like interface and `fit`, `predict`, `predict_proba` with scikit-learn-like wrappers.
- Portable models: Save/load, and export to ONNX for broad runtime support.
- Extensible design: A common trait-based core for trees, forests, and boosted ensembles.

# Project layout
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
│   ├── data                   — Arrow-backed data access & dataset abstractions
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

# Features

- CART decision trees (classification & regression)
    - Gini / Entropy (cls), MSE / MAE (reg), max depth, min samples, etc.
    - Missing values & categorical support (Arrow DictionaryArray)
- Batch prediction (multi-threaded), predict_proba for classifiers
- Model persistence (serde JSON) and ONNX export (TreeEnsemble*)
- Interoperability:
    - Accepts NumPy, pandas, Polars, PyArrow inputs
    - Returns NumPy arrays (predictions), Arrow tables where appropriate
- Performance knobs: multi-threading, optional histogram splitters (planned)
- Compiled inference (planned): treelite/LLVM-style codegen backend

# Installation
## Prerequisites

Rust (stable, latest) and Cargo

Python >= 3.12

Recommended Python deps (installed automatically by wheels or when building from source):

- numpy>=1.26
- pyarrow>=12
- pandas>=2.0 (optional)
- polars>=1.0 (optional)

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
# Build a simple dense dataset. Features are currently ignored by the model,
# but must match the number of target values.
import numpy as np
import pandas as pd

from forestfire import TargetMeanTree

X = pd.DataFrame({"x1": [0,1,1,0], "x2": [1,1,0,0]})
y = np.array([0,1,1,0])

clf = TargetMeanTree.fit(X, y)   # accepts pandas/NumPy/Polars/PyArrow
pred  = clf.predict(X)           # -> NumPy array [n_samples]

# Persistence & ONNX export, still not implemented
clf.save("tree.onnx")
loaded = TargetMeanTree.load("tree.onnx")
```

# Data interop tips

pandas -> Arrow zero-copy happens under the hood (via Arrow C Data Interface).

Polars DataFrame is supported (via pyo3-polars); you can also pass `df.to_arrow()` directly.

Using NumPy? We accept ndarray for X and y, converting efficiently.

# Quickstart (Rust)
```rust
use anyhow::Result;
use forestfire_data::DenseDataset;
use forestfire_core::tree::TargetMeanTree;

fn main() -> Result<()> {
    // Build a simple dense dataset. Features are currently ignored by the model,
    // but must match the number of target values.
    let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let y = vec![10.0, 12.0, 14.0, 20.0];
    let ds = DenseDataset::new(x, y)?;            // validates shapes

    // Train the "target mean" tree: stores global mean(y) and predicts it.
    let model = TargetMeanTree::train(&ds)?;      // error if empty/mismatched lengths

    // Predict a constant for every row in the dataset (mean of y).
    let preds = model.predict_dataset(&ds);       // -> Vec<f64> of length n_samples
    println!("{preds:?}");                        // [14.0, 14.0, 14.0, 14.0]

    // Persistence & ONNX export, still not implemented
    model.save("tree.onnx")
    let loaded = TargetMeanTree.load("tree.onnx")
    Ok(())
}
```

Heavy loops release the Python GIL; training & prediction are multi-threaded by default.

# Design notes
- Arrow everywhere: The core stores features in Arrow arrays for columnar, cache-friendly scans.
- Numerical and categorical features (Arrow DictionaryArray) supported.
- Missing values tracked via Arrow validity bitmaps.
- Traits & reuse: Predictor/Trainable traits unify trees/ensembles. Split finding logic is shared.
- Memory: Node arrays and compact types (e.g., f32 thresholds) keep models lean; future array-based layout optimizes cache locality further.
- Parallelism: Feature-wise split evaluation and tree/forest training are parallelized with Rayon.

# Persistence & interchange
- Save/Load: JSON via serde for human-readable model storage.
- ONNX export: Emits TreeEnsembleClassifier / TreeEnsembleRegressor; loadable in ONNX Runtime and many other environments.

# Roadmap
## Immediate Priorities
- Backend to `DenseDataset`
- Exact CART algorithm
- Splitting criterion
- Histogram CART algorithm
- Compiled inference backend (`treelite` / LLVM-style codegen) for inference speedups

## Long term Priorities
- Random Forest (bagging, feature subsampling)
- Gradient Boosted Trees (with histogram-based split finding)
- Out-of-core training via Arrow streaming/memory-mapping
- Calibrated probabilities, monotonic constraints, categorical split optimizations
- Sklearn estimator wrappers (get_params, set_params, sklearn compat tests)

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
No. You can pass NumPy arrays directly. pandas/Polars/pyarrow are supported for zero-copy convenience.

## Q: Are predictions deterministic?
Yes for single trees given the same inputs/params. For randomized procedures (e.g., forests), set `random_state`.

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
