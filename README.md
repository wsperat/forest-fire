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

Python 3.9–3.12

Recommended Python deps (installed automatically by wheels or when building from source):

numpy>=1.26, pandas>=2.0, pyarrow>=12, polars>=1.0 (optional)

Option A — from source with maturin
# 1) In your Python environment
pip install maturin

# 2) Build & install the Python package in editable/dev mode
maturin develop -m python-bindings/Cargo.toml

# or build a wheel
maturin build -m python-bindings/Cargo.toml --release
pip install dist/*.whl

Option B — Rust only
# Use the Rust core crate directly in your Cargo project
cargo add core --git https://github.com/<org>/<repo>  # adjust URL/name

Quickstart (Python)
import numpy as np
import pandas as pd

# The module name is produced by python-bindings; replace if you changed it.
from tree_ml import DecisionTreeClassifier, DecisionTreeRegressor

# Classification
X = pd.DataFrame({"x1": [0,1,1,0], "x2": [1,1,0,0]})
y = np.array([0,1,1,0])

clf = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=42)
clf.fit(X, y)                    # accepts pandas/NumPy/Polars/PyArrow
proba = clf.predict_proba(X)     # -> NumPy array [n_samples, n_classes]
pred  = clf.predict(X)           # -> NumPy array [n_samples]

# Persistence & ONNX export
clf.save("tree.json")
loaded = DecisionTreeClassifier.load("tree.json")
loaded.export_onnx("tree.onnx")


Data interop tips

pandas → Arrow zero-copy happens under the hood (via Arrow C Data Interface).

Polars DataFrame is supported (via pyo3-polars); you can also pass df.to_arrow() directly.

Still on NumPy? We accept ndarray for X and y, converting efficiently.

Quickstart (Rust)
use core::data::Dataset;
use core::tree::{DecisionTree, SplitCriterion};

fn main() -> anyhow::Result<()> {
    // Build a dataset from Arrow arrays (see core::data helpers)
    let ds = Dataset::from_iter(vec![
        ("x1", vec![0.0, 1.0, 1.0, 0.0]),
        ("x2", vec![1.0, 1.0, 0.0, 0.0]),
    ])?;
    let y = vec![0_i32, 1, 1, 0];

    let mut tree = DecisionTree::classifier()
        .max_depth(3)
        .criterion(SplitCriterion::Gini);

    tree.fit(&ds, &y)?;
    let preds = tree.predict(&ds)?;
    println!("{preds:?}");
    Ok(())
}

Python API surface (first release)
DecisionTreeClassifier(
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: int | float | str | None = None,
    criterion: str = "gini",          # "gini" | "entropy"
    random_state: int | None = None,
)

DecisionTreeRegressor(
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: int | float | str | None = None,
    criterion: str = "mse",           # "mse" | "mae"
    random_state: int | None = None,
)

# Common methods
.fit(X, y)              # X: numpy/pandas/polars/pyarrow; y: numpy/pandas/pyarrow
.predict(X) -> np.ndarray
.predict_proba(X) -> np.ndarray          # classifiers
.save(path) / .load(path)                # JSON via serde
.export_onnx(path)                       # ONNX-ML TreeEnsemble*


Heavy loops release the Python GIL; training & prediction are multi-threaded by default.

Design notes

Arrow everywhere: The core stores features in Arrow arrays for columnar, cache-friendly scans.

Numerical and categorical features (Arrow DictionaryArray) supported.

Missing values tracked via Arrow validity bitmaps.

Traits & reuse: Predictor/Trainable traits unify trees/ensembles. Split finding logic is shared.

Memory: Node arrays and compact types (e.g., f32 thresholds) keep models lean; future array-based layout optimizes cache locality further.

Parallelism: Feature-wise split evaluation and tree/forest training are parallelized with Rayon.

Persistence & interchange

Save/Load: JSON via serde for human-readable model storage.

ONNX export: Emits TreeEnsembleClassifier / TreeEnsembleRegressor; loadable in ONNX Runtime and many other environments.

Roadmap

Random Forest (bagging, feature subsampling)

Gradient Boosted Trees (with histogram-based split finding)

Compiled inference backend (treelite / LLVM-style codegen) for 10×+ speedups

Out-of-core training via Arrow streaming/memory-mapping

Calibrated probabilities, monotonic constraints, categorical split optimizations

Sklearn estimator wrappers (get_params, set_params, sklearn compat tests)

Building & testing
Rust
# Lint & test
cargo fmt --all
cargo clippy --all -- -D warnings
cargo test --workspace

# Benchmarks (Criterion)
cargo bench

Python
# Build locally for your venv
maturin develop -m python-bindings/Cargo.toml

# Run Python tests (if present)
pytest -q

Versioning & compatibility

Rust: stable toolchain (latest two releases supported)

Python: 3.9–3.12 (x86_64 & aarch64 on Linux/macOS/Windows)

Binary wheels: built via maturin; source installs require Rust toolchain

Contributing

Contributions are welcome! Please:

Open an issue describing the change/feature.

Add tests and benchmarks for performance-sensitive code.

Run cargo fmt, clippy, cargo test, and pytest before submitting.

See CONTRIBUTING.md (or open an issue if you don’t see one yet).

Acknowledgments

PyO3 for seamless Python extensions in Rust

Apache Arrow for columnar speed & zero-copy interop

NumPy / pandas / Polars communities for foundational data tooling

ONNX-ML for portable, fast inference formats

FAQ

Q: Do I need pandas or Polars?
No. You can pass NumPy arrays directly. pandas/Polars/pyarrow are supported for zero-copy convenience.

Q: How do I disable multi-threading?
Set the Rayon global thread pool via RAYON_NUM_THREADS=1 (or configure the pool in Rust).

Q: Are predictions deterministic?
Yes for single trees given the same inputs/params. For randomized procedures (e.g., forests), set random_state.

Q: How big can my data be?
As large as your machine’s memory (for now). Out-of-core/streaming training is on the roadmap.

MIT License

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
