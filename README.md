# ForestFire
![ForestFire](docs/forest-fire.jpg)

## Fast tree-based learning in Rust

ForestFire is a tree-learning library with a Rust core and a Python API. The current implementation is centered around three ideas:

- one unified `train(...)` interface instead of learner-specific entrypoints
- one unified training `Table` abstraction that chooses the right internal layout automatically
- one explicit JSON IR for serialization, portability, and future runtime/export work

## What exists today

- Unified `train` API in Rust and Python
- Automatic table selection between `DenseTable` and `SparseTable`
- Classification trees: `id3`, `c45`, `cart`, `oblivious`
- Regression trees: `cart`, `randomized`, `oblivious`
- Criterion selection via `gini`, `entropy`, `mean`, `median`, or `auto`
- Canary-based automatic growth stopping
- Physical-core-aware parallel training
- Optimized inference runtimes via `optimize_inference(...)`
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
    bins="auto",
    physical_cores=4,
)

preds = model.predict(X)
fast_model = model.optimize_inference(physical_cores=4)
fast_preds = fast_model.predict(X)
compiled = fast_model.serialize_compiled()
restored_fast = forestfire.OptimizedModel.deserialize_compiled(
    compiled, physical_cores=4
)
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
            bins: NumericBins::Auto,
            physical_cores: Some(4),
        },
    )?;

    let preds = model.predict_rows(x.clone())?;
    let fast_model = model.optimize_inference(Some(4))?;
    let fast_preds = fast_model.predict_rows(x.clone())?;
    let serialized = model.serialize_pretty()?;
    let restored = forestfire_core::Model::deserialize(&serialized)?;

    assert_eq!(preds, fast_preds);
    assert_eq!(preds, restored.predict_rows(x)?);
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
4. use `predict(...)` on raw inference data
5. optionally call `optimize_inference(...)` when scoring is performance-critical
6. serialize the result to the JSON IR when you need a portable artifact

That is why the API is organized around `Table`, `train`, `predict`, `optimize_inference`, and `serialize`, rather than around many learner-specific classes.

## Training interface

Python:

```python
train(
    X_or_table,
    y=None,
    algorithm="dt",
    task="auto",
    tree_type="cart",
    criterion="auto",
    canaries=2,
    bins="auto",
    physical_cores=None,
    n_trees=None,
    max_features=None,
    seed=None,
    compute_oob=False,
)
```

Rust:

```rust
train(&table, TrainConfig { ... })
```

### Parameters

#### `algorithm`

Current values:

- `dt`
- `rf`

Why it exists:

- It keeps the top-level API stable as the library grows beyond a single learner family.
- It separates the learner family from the exact tree structure.

Why it is a string in Python but an enum in Rust:

- Python gets a familiar LightGBM-style interface.
- Rust keeps typed validation and exhaustive dispatch.

#### `task`

Current values:

- `auto`
- `regression`
- `classification`

Why it exists:

- `auto` infers classification for integer, boolean, and string targets, and regression for float targets.
- Task choice changes leaf semantics, scoring, defaults, and supported tree types.

#### `tree_type`

Current support:

- regression: `cart`, `randomized`, `oblivious`
- classification: `id3`, `c45`, `cart`, `randomized`, `oblivious`

Why it exists:

- Tree structure is one of the most meaningful modeling decisions.
- Users should choose the family explicitly instead of inferring it indirectly from many secondary knobs.

What each one means:

- `id3`: entropy-driven multiway-style classifier
- `c45`: practical extension of ID3
- `cart`: conventional binary tree
- `randomized`: stochastic split-search variant
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
- `cart`, `randomized`, `oblivious` classification -> `gini`
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

#### `bins`

Current values:

- `"auto"` in Python / `NumericBins::Auto` in Rust
- integer `1..=128` in Python / `NumericBins::Fixed(usize)` in Rust

Why it exists:

- split search gets much cheaper once numeric features are mapped into a bounded bin space
- power-of-two bin counts fit the optimized inference kernels cleanly
- the right bin budget depends on how much distinct signal a feature actually has

Current `auto` behavior:

- for each numeric feature, ForestFire chooses the highest power of two up to `128`
- the chosen count is capped by the number of distinct observed values
- that keeps every realized bin populated while still giving the learner as much split resolution as the data supports

Why that is the default:

- small datasets do not waste work on hundreds of empty bins
- larger datasets can still climb up to `128` bins
- the resulting representation stays regular for both training and optimized prediction

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

#### `compute_oob`

Default:

- `False`

Why it exists:

- Random forests can estimate generalization quality from out-of-bag rows without needing a separate validation split.

Behavior:

- only meaningful for `algorithm="rf"`
- classification forests expose OOB accuracy
- regression forests expose OOB `R^2`
- non-forest models always report `compute_oob=False` and `oob_score=None`

## Tree introspection

Tree models expose a small introspection API on both `Model` and `OptimizedModel`.

Available members:

- `tree_count`
- `tree_structure(tree_index=0)`
- `tree_prediction_stats(tree_index=0)`
- `tree_node(node_index, tree_index=0)` for standard trees
- `tree_level(level_index, tree_index=0)` for oblivious trees
- `tree_leaf(leaf_index, tree_index=0)` for all trees

What it is for:

- understanding realized tree shape after training
- inspecting cutoffs and leaf payloads
- summarizing the distribution of prediction values stored in leaves
- inspecting one tree at a time inside a forest

What `tree_structure(...)` returns:

- `representation`: `"node_tree"` or `"oblivious_levels"`
- `node_count`
- `internal_node_count`
- `leaf_count`
- `actual_depth`
- `shortest_path`
- `longest_path`
- `average_path`

What `tree_prediction_stats(...)` returns:

- `count`
- `unique_count`
- `min`
- `max`
- `mean`
- `std_dev`
- `histogram`

How it works:

- introspection is derived from the semantic IR, not from a separate debug-only format
- standard trees are inspected through exported nodes
- oblivious trees are inspected through exported levels and leaves
- optimized models delegate to the same semantic model, so introspection matches the original model exactly
- forests use `tree_index` to inspect a specific constituent tree

Python example:

```python
model = train(X, y, task="classification", tree_type="cart")

summary = model.tree_structure()
stats = model.tree_prediction_stats()
root = model.tree_node(0)
leaf = model.tree_leaf(0)

print(summary["leaf_count"], summary["actual_depth"])
print(stats["histogram"])
print(root["kind"])
print(leaf["leaf"])
```

## Table system

Training always goes through a `Table`. That layer is responsible for validation, preprocessing, and choosing the backing memory layout.

Inference is different: users are expected to pass raw rows, named columns, arrays, dataframes, or sparse matrices directly to `predict(...)`. `Table` is primarily the training-side abstraction.

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
- numeric rank-binning into a power-of-two bin count, using the highest populated count up to `128` by default
- canaries built at the table layer so every learner sees the same stopping reference

Why the numeric binning is rank-based:

- tree splits mostly care about order, not absolute scale
- rank-based bins are stable under monotonic rescaling
- it gives a compact, bounded representation for split search
- adaptive power-of-two counts avoid empty-bin overhead on small or low-cardinality features

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

## Inference optimization

ForestFire now exposes a second inference form:

- Rust: `model.optimize_inference(Some(physical_cores))?`
- Python: `model.optimize_inference(physical_cores=...)`

This produces an optimized runtime object that preserves exactly the same semantics, predictions, and JSON IR as the original model. The optimized object is prediction-only. Serialization still comes from the same underlying model representation, which is why optimized and non-optimized models export the same IR.

### What is optimized

The optimized runtime does a few different things:

- CART-style binary trees are lowered into prediction-only fallthrough/jump layouts
- binary splits use array-indexed or fallthrough child selection instead of repeated training-structure branching
- multiway classifier splits precompute a dense bin-to-child lookup table, replacing per-row linear search over branches
- oblivious trees are stored as compact per-level feature and threshold arrays, then evaluated by accumulating the leaf index directly
- multi-row inputs are preprocessed together before scoring
- compiled binary and oblivious runtimes convert those rows into compact column-major binned matrices so one split can scan many rows in one pass
- `LazyFrame` inputs are streamed in batches of about `10_000` rows instead of being collected into one giant materialized dataframe first
- batch prediction is parallelized across rows with a dedicated Rayon thread pool created by `optimize_inference(...)`
- those batch columns are stored as `u8` whenever a feature’s effective bins fit in `<= 255`, and as `u16` only when larger bin ids are actually needed

### How each optimization affects low-level compute

#### 1. Prediction-only node layouts

Training trees carry more structure than inference needs:

- training statistics
- richer enum variants
- fields used for export or debugging

The optimized runtime strips that down to the minimum prediction state:

- leaf value
- feature index
- threshold bin
- child locations

At the CPU level this reduces:

- pointer chasing through larger objects
- cache pressure from unused fields
- branch work spent unpacking more general training-time representations

In practice, that matters most for standard trees, where every scored row walks a chain of nodes and repeatedly touches that node memory.

#### 2. Compiled CART-style fallthrough traversal

For binary trees, the optimized runtime lowers each node into a compact branch record:

- feature index
- threshold bin
- one jump target
- one “jump when predicate is true/false” flag

The more common child is laid out as the next node in memory, and the less common child becomes the explicit jump target. That is the same general idea used by highly tuned tree runtimes: keep the hot branch as fallthrough and pay the jump only for the less common path.

At the CPU level this reduces:

- child-pointer loads
- node size
- unpredictable control flow in the hot loop
- wasted instruction-cache space on training-only structure

#### 3. Dense lookup for multiway splits

Multiway classifier nodes are especially unfriendly to low-level prediction speed if implemented literally:

- read the row’s bin
- scan the branch list
- compare against each branch bin until a match is found

The optimized runtime replaces that with:

- read the row’s bin
- index a precomputed `bin -> child` table

That changes the cost from “a small search with repeated comparisons” to “one bounds-known array access”. Low-level effects:

- fewer dependent comparisons
- fewer unpredictable loop exits
- more regular memory access
- easier vectorizer and prefetcher behavior, even if the full traversal is still branchy overall

This optimization matters most for `id3` and `c45` style trees, where multiway nodes are common.

#### 4. Compact oblivious-tree execution

Oblivious trees are already structurally friendly to fast inference because every depth uses the same split shape. The optimized runtime makes that explicit:

- one array of feature indices
- one array of threshold bins
- one leaf array

Prediction then becomes:

- iterate levels in order
- compute one bit per level
- build the final leaf index
- load the leaf value once

At the hardware level this is much more regular than standard-tree traversal:

- the loop body is fixed
- memory accesses are predictable
- no child-pointer chasing is needed
- leaf selection becomes a final indexed load

That is why oblivious trees are usually the easiest tree family to push toward very high inference throughput.

#### 5. Whole-batch preprocessing and compact column-major binned matrices

The optimized runtime preprocesses multi-row inputs as a batch before traversal. For the compiled binary and oblivious kernels, those batches are then rearranged into compact column-major bin matrices.

Why that helps:

- one split can scan the same feature column across many rows
- repeated threshold checks hit compact integer buffers instead of raw `f64` values
- bin ids are much smaller than raw `f64` values, so more working set fits in cache
- rows can be partitioned in-place by branch outcome without re-reading the full input object model

With adaptive binning, this is now tighter than before:

- features with small effective bin domains are packed as `u8`
- only features that actually exceed `255` need `u16`
- compact columns reduce memory bandwidth and improve cache density in the hot loops

This especially helps when:

- there are many rows
- compiled binary trees touch multiple splits per prediction
- raw input coercion is already done and traversal becomes the main cost

#### 6. Batch partitioning for compiled binary trees

For compiled CART-style trees, the optimized runtime does not walk each row independently through the full tree. Instead, it keeps a mutable row-index buffer for a batch chunk and partitions that buffer at each node into:

- rows that fall through
- rows that jump

Then each child subtree works on its own contiguous row segment.

Low-level effect:

- one feature column is read across many rows before moving on
- branch decisions become data partition work rather than “full tree traversal per row”
- the predictor touches fewer unrelated cache lines at once

This is the closest part of the runtime to the “compiled tree evaluation” style used by tools like Treelite and lleaves: not source-code generation, but a lowered execution form designed around whole-batch traversal rather than the original training objects.

#### 7. Row-parallel batch scoring

Batch prediction is embarrassingly parallel across rows. `optimize_inference(...)` therefore builds a dedicated inference thread pool and parallelizes over rows once the batch is large enough.

Low-level effect:

- each worker thread traverses the same model on different rows
- model memory is shared read-only
- per-row state stays thread-local

That is a good CPU pattern because:

- synchronization is minimal
- there is no write contention in the hot path
- scaling is usually limited by memory hierarchy and branch behavior rather than lock overhead

This optimization helps most on larger batches. On tiny batches, thread scheduling and setup costs can dominate.

#### 8. LazyFrame streaming

`polars.LazyFrame` inputs are treated differently from already materialized arrays/dataframes:

- the lazy query is sliced into batches of about `10_000` rows
- each batch is collected
- that batch is preprocessed and scored
- predictions are appended in order

This avoids materializing an arbitrarily large lazy result all at once while still letting each collected batch benefit from the same compiled/batched predictor path.

### Why this is faster

These optimizations target the main sources of inference cost:

- branch prediction misses from irregular tree traversal
- pointer chasing through training-oriented tree structures
- repeated branch scans for multiway splits
- poor memory locality when scoring many rows
- leaving CPU cores idle on embarrassingly parallel batch prediction

The optimized runtime does not change the model. It changes only how that same model is laid out and executed.

### Where it helps the most

The biggest gains usually come from:

- large prediction batches, where row-parallel execution can keep all requested cores busy
- deeper standard trees, where branch-reduced traversal matters more
- compiled binary trees, where fallthrough layout and batch partitioning reduce traversal overhead
- classifier trees with multiway nodes, because the dense child lookup avoids repeated branch search
- repeated scoring workloads, where the one-time optimization cost is amortized across many predictions
- dense numeric or mixed tabular batches, where traversal cost dominates once preprocessing has been done
- large lazyframe predictions, where `10_000`-row batching avoids full eager materialization

### Where it helps less

The gains are usually smaller for:

- tiny prediction batches, where thread-pool and preprocessing overhead dominate
- very shallow trees, where there is little branch structure to optimize away
- cases where input coercion dominates total latency more than model traversal does
- oblivious single-core scoring, where the extra batch-layout conversion can still outweigh the traversal savings on some workloads

### Why this also matters for future GPU work

The optimized runtime is intentionally flatter than the training structures:

- standard trees become simple prediction nodes
- oblivious trees become level arrays plus leaf arrays
- inference inputs become compact binned matrices

That is a better starting point for future compiled CPU kernels and GPU kernels than the richer training-time representation. The IR does not need to change for that work, because the optimized runtime is just another execution strategy for the same exported model.

## Serialization and IR

Trained models can be:

- serialized to a JSON string
- deserialized back into a runnable model
- exported as the explicit IR JSON
- optimized and serialized into a compiled CPU artifact for fast reload

Rust:

- `model.serialize()`
- `model.serialize_pretty()`
- `Model::deserialize(...)`
- `model.to_ir()`
- `model.to_ir_json()`
- `model.to_ir_json_pretty()`
- `Model::json_schema_json_pretty()`
- `optimized.serialize_compiled()`
- `OptimizedModel::deserialize_compiled(bytes, Some(physical_cores))`

Python:

- `model.serialize(pretty=False)`
- `Model.deserialize(serialized)`
- `model.to_ir_json(pretty=False)`
- `optimized.serialize_compiled()`
- `OptimizedModel.deserialize_compiled(serialized, physical_cores=None)`

### Compiled optimized artifacts

Optimized CPU runtimes now have a second serialization path: a compiled artifact.

This artifact is intentionally different from the JSON IR:

- the JSON IR is the canonical semantic model
- the compiled artifact is a versioned binary snapshot of the already-lowered optimized CPU runtime

The compiled artifact stores:

- a fixed magic/version/backend header
- the semantic model IR payload needed to preserve the exact same exported model meaning
- the compiled runtime layout, so reload does not have to lower the semantic model into prediction nodes again

This mirrors the general split used by high-performance inference systems:

- one representation for portability and inspection
- one representation for fast runtime loading

In ForestFire, the compiled artifact is currently:

- CPU-only
- versioned
- binary
- backend-specific by design

It does not replace the JSON IR. It is an execution artifact for optimized runtimes.

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

- [crates/core/schema/forestfire-ir.schema.json](crates/core/schema/forestfire-ir.schema.json)

It is generated from the Rust IR types and checked in as a contract artifact. The test suite verifies that the generated schema matches the checked-in file exactly.

## Support matrix

### Tasks and tree types

- `task="regression"` with `tree_type="cart" | "randomized" | "oblivious"`
- `task="classification"` with `tree_type="id3" | "c45" | "cart" | "randomized" | "oblivious"`

### Python input types

- NumPy arrays
- Python sequences
- pandas
- polars
- pyarrow
- SciPy dense matrices
- SciPy sparse matrices

## Benchmarks

Run:

```bash
task benchmark-inference
```

Artifacts are written to:

- [docs/benchmarks/inference_benchmark_results.json](docs/benchmarks/inference_benchmark_results.json)
- [docs/benchmarks/cart_runtime.png](docs/benchmarks/cart_runtime.png)
- [docs/benchmarks/cart_speedup.png](docs/benchmarks/cart_speedup.png)
- [docs/benchmarks/oblivious_runtime.png](docs/benchmarks/oblivious_runtime.png)
- [docs/benchmarks/oblivious_speedup.png](docs/benchmarks/oblivious_speedup.png)

Current checked-in run:

- compiled CART optimized single-core averages about `1.056x` baseline
- compiled CART optimized parallel averages about `1.053x` baseline
- oblivious optimized single-core averages about `0.902x` baseline
- oblivious optimized parallel averages about `1.009x` baseline

That matches the current runtime:

- compiled CART benefits strongly from fallthrough layout, batch partitioning, and compact `u8`/`u16` bins
- oblivious trees improve mainly when batching and parallelism amortize the batch-conversion cost

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
