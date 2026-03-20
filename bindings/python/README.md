# forestfire (Python)

Python bindings for the unified ForestFire training interface.

Long-running training, optimized-runtime construction, and heavy prediction work release the GIL
before entering the Rust hot path.

The Python package is built around five objects:

- `Table` for validated training data
- `train(...)` for fitting
- `Model.predict(...)` for inference
- `Model.predict_proba(...)` for classification probabilities
- `Model.optimize_inference(...)` for optimized inference runtimes
- `Model.serialize(...)` / `Model.deserialize(...)` for portability

## Quickstart

```python
import numpy as np

from forestfire import Table, train

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
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
proba = model.predict_proba(X)
fast_model = model.optimize_inference(physical_cores=4)
fast_preds = fast_model.predict(X)
fast_proba = fast_model.predict_proba(X)
compiled = fast_model.serialize_compiled()
restored_fast = fast_model.deserialize_compiled(compiled, physical_cores=4)
serialized = model.serialize(pretty=True)
restored = model.deserialize(serialized)
ir_json = model.to_ir_json(pretty=True)
```

## Training API

```python
train(
    x,
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
    learning_rate=None,
    bootstrap=False,
    top_gradient_fraction=None,
    other_gradient_fraction=None,
)
```

### Current supported values

- `algorithm="dt" | "rf" | "gbm"`
- `task="auto" | "regression" | "classification"`
- `tree_type="id3" | "c45" | "cart" | "randomized" | "oblivious"`
- `criterion="auto" | "gini" | "entropy" | "mean" | "median"`

### Why these parameters exist

#### `algorithm`

`algorithm` selects the learner family:

- `dt`: one tree
- `rf`: bagged ensemble with bootstrap sampling and feature subsampling
- `gbm`: stage-wise second-order boosting with shrinkage and gradient-focused row sampling

#### `task`

`task="auto"` infers classification for integer, boolean, and string targets, and regression for float targets. An explicit task still controls split scoring, leaf semantics, defaults, and the set of valid tree types.

#### `tree_type`

`tree_type` selects the structural family directly:

- `id3`: entropy-first classifier
- `c45`: practical extension of ID3
- `cart`: standard binary tree
- `randomized`: stochastic split-search variant
- `oblivious`: symmetric tree with one split per depth

#### `criterion`

Criterion changes the model itself, not just training speed:

- `gini` and `entropy` for classification
- `mean` and `median` for regression
- `auto` to pick the default implied by task and tree type

Current `auto` behavior:

- `id3`, `c45` -> `entropy`
- classification `cart`, `randomized`, `oblivious` -> `gini`
- regression models -> `mean`
- `gbm` trains second-order trees internally when `criterion="auto"`

#### `canaries`

ForestFire uses automatic growth stopping instead of pruning. Canary variables are shuffled copies of the already-preprocessed features. If a learner chooses a canary, it has reached a noise-like split and stops growing there.

Current stopping behavior:

- standard trees stop at the current node
- oblivious trees stop the remaining depth growth
- `gbm` stops adding new stages when the first/root split that would be taken is a canary

#### `bins`

Current values:

- `"auto"`
- integer `1..=128`

Why it exists:

- split search benefits from bounded numeric cardinality
- power-of-two bin counts fit the optimized runtimes cleanly
- different datasets need different bin budgets

Current `auto` behavior:

- per numeric feature, ForestFire picks the highest power of two up to `128`
- the chosen count is also capped by the number of distinct observed values
- that keeps every realized bin populated while avoiding a larger-than-useful bin space

Why this is the default:

- tiny datasets do not waste work on hundreds of empty bins
- larger datasets can still use up to `128` bins
- the result stays regular enough for fast training and inference kernels

#### `physical_cores`

This controls CPU usage during fitting. The library uses physical cores as the public knob because split scoring is memory-sensitive and that tends to be a more honest resource limit than logical threads.

#### `compute_oob`

This is only meaningful for `algorithm="rf"`.

When enabled, the forest keeps track of each tree’s out-of-bag rows and exposes:

- `model.compute_oob`
- `model.oob_score`

Current meaning of `oob_score`:

- classification: OOB accuracy
- regression: OOB `R^2`

#### `learning_rate`

This is only meaningful for `algorithm="gbm"`.

- Each stage prediction is multiplied by `learning_rate` before it is added to the ensemble.
- Lower values generally need more trees.

#### `bootstrap`

This is only meaningful for `algorithm="gbm"`.

- `False`: each stage starts from the full table, then applies gradient-focused row sampling.
- `True`: each stage first draws a bootstrap sample, then applies gradient-focused row sampling.

#### `top_gradient_fraction` and `other_gradient_fraction`

These are only meaningful for `algorithm="gbm"`.

- The booster uses a LightGBM-style gradient-focused sampler.
- `top_gradient_fraction` keeps the largest-gradient rows.
- `other_gradient_fraction` samples additional rows from the remainder.

## Tables and input handling

### `Table`

`Table` is the public container for validated training data. You can pass raw data directly to `train(...)`, but building a `Table` explicitly is useful when you want preprocessing and validation separated from fitting.

`Table` is intentionally training-oriented. For inference, the intended path is to pass raw arrays, dicts, dataframes, lazyframes, or sparse matrices directly to `predict(...)` or `predict_proba(...)`.

`Table` chooses between:

- `DenseTable` for mixed numeric/binary data
- `SparseTable` for binary sparse inputs

### Supported input types

- NumPy arrays
- Python sequences
- pandas
- polars
- pyarrow
- SciPy dense matrices
- SciPy sparse matrices

### `DenseTable`

`DenseTable` is Arrow-backed and optimized for repeated feature scans. Numeric features are rank-binned into a power-of-two number of bins, using the highest populated count up to `128` by default, and binary `0/1` columns are stored as booleans so they are both smaller and cheaper to split on.

### `SparseTable`

`SparseTable` is binary-only. Internally it stores, per feature, the row positions where the value is `1`. That keeps memory proportional to the number of positive entries rather than the full dense shape.

SciPy sparse matrices are converted into this representation by reading their shape and nonzero coordinates. They are not densified first.

## Optimized inference

Use:

- `fast_model = model.optimize_inference(physical_cores=None)`

The returned `OptimizedModel` predicts the same values and serializes to the same IR as the original `Model`. It is a prediction-optimized runtime view over the same trained model, not a different model artifact.

### What it changes internally

- CART-style binary trees are lowered into compact fallthrough/jump layouts
- binary splits pick the next child by fallthrough or one stored jump instead of the original training structure
- multiway classifier splits use a dense bin lookup table instead of scanning the branch list
- oblivious trees are evaluated from compact level arrays into a leaf index
- multi-row inputs are preprocessed together before scoring
- compiled binary and oblivious runtimes use compact column-major binned matrices so one split can scan many rows at once
- `polars.LazyFrame` inputs are collected and scored in batches of about `10_000` rows
- row batches are scored in parallel across the requested physical cores
- batch columns are stored as `u8` whenever a feature’s effective bins fit in `<= 255`, and as `u16` only when larger bin ids are actually needed

### What that means at the CPU level

#### Prediction-only node layouts

The optimized runtime removes training-only fields from the hot prediction path. That reduces object size, cache pressure, and the amount of general-case logic the predictor loop has to carry around.

#### Compiled CART-style fallthrough layout

For binary trees, the optimized runtime keeps the more common child as the next node in memory and stores only the less common branch as an explicit jump target. That shrinks the hot node representation and reduces branch-heavy traversal logic.

#### Dense lookup for multiway nodes

For multiway classifier nodes, the optimized runtime replaces “scan branches until one matches” with “index the precomputed child table by bin id”. That reduces dependent comparisons and makes access more regular.

#### Compact oblivious-tree loops

Oblivious trees become a sequence of feature-index and threshold arrays plus a final leaf array. Scoring becomes a fixed loop that accumulates a leaf index, which is much more regular than standard pointer-chasing tree traversal.

#### Whole-batch preprocessing and compact column-major batches

Inference inputs are converted into compact bin ids before the optimized traversal runs. For compiled binary and oblivious runtimes, those bins are arranged column-major so the predictor can read one feature across many rows before moving on.

With adaptive binning, the batch layout is tighter than before:

- features with small effective bin domains are packed as `u8`
- only features that actually exceed `255` need `u16`
- compact columns reduce memory bandwidth and improve cache density in the hot loops

#### Batch partitioning for compiled binary trees

Compiled CART-style trees operate on row batches, not just one row at a time. At each node, the runtime partitions a row-index buffer into “fallthrough” and “jump” segments, then continues traversal on those contiguous row slices. That keeps feature reads and branch outcomes grouped together.

#### Row-parallel scoring

Rows are independent at prediction time, so the optimized runtime parallelizes across them with a dedicated thread pool. That keeps the model read-only and shared while each worker operates on separate rows.

#### LazyFrame streaming

`polars.LazyFrame` inputs are not collected all at once. They are sliced into batches of roughly `10_000` rows, collected, preprocessed, scored, and appended in order. That keeps memory usage bounded while still benefiting from the optimized batch kernels.

### Where it helps most

- large prediction batches
- deeper trees
- compiled binary trees
- multiway classifiers
- repeated scoring of the same model
- large lazyframe predictions

### Where it helps less

- tiny batches
- very shallow trees
- workloads dominated by input conversion instead of traversal
- some single-core oblivious workloads, where layout conversion can still dominate

### Why the IR stays the same

`OptimizedModel` delegates serialization and IR export to the original semantic model. That keeps portability stable: optimization changes execution strategy, not model meaning.

## Tree introspection

Both `Model` and `OptimizedModel` expose:

- `tree_count`
- `tree_structure(tree_index=0)`
- `tree_prediction_stats(tree_index=0)`
- `tree_node(node_index, tree_index=0)` for standard trees
- `tree_level(level_index, tree_index=0)` for oblivious trees
- `tree_leaf(leaf_index, tree_index=0)` for all trees
- `to_dataframe(tree_index=None)`

These methods work for:

- standalone decision trees
- forests via `tree_index`
- optimized models, which delegate to the same semantic tree information as the base model

### `to_dataframe(...)`

Returns a polars `DataFrame` when `polars` is installed. If not, it falls back to a `pyarrow.Table`.

- standard trees produce rows for split nodes, leaf nodes, and unmatched fallback leaves on multiway splits
- oblivious trees produce rows for each level and each leaf
- forests include a `tree_index` column, and `tree_index=...` filters to one constituent tree

Common columns include:

- `tree_index`
- `representation`
- `node_type`
- `node_index`
- `depth`
- `parent_index`
- `split_feature`
- `split_feature_name`
- `split_type`
- `threshold_bin`
- `threshold_upper_bound`
- `operator`
- `left_child`
- `right_child`
- `branch_bins`
- `branch_children`
- `leaf_value`
- `leaf_class_index`
- `leaf_label`
- `sample_count`
- `impurity`
- `gain`
- `variance`
- `class_counts`

This is meant for inspection and analysis workflows that are easier in tabular form, similar to LightGBM's tree dataframe export.

### `tree_structure(...)`

Returns a Python `dict` with:

- `representation`
- `node_count`
- `internal_node_count`
- `leaf_count`
- `actual_depth`
- `shortest_path`
- `longest_path`
- `average_path`

For standard trees, the path metrics are measured from the root to the leaves actually present after training. For oblivious trees, every leaf is at the same depth, so shortest, longest, and average path are identical.

### `tree_prediction_stats(...)`

Returns a Python `dict` with:

- `count`
- `unique_count`
- `min`
- `max`
- `mean`
- `std_dev`
- `histogram`

`histogram` is a list of `{ "prediction": ..., "count": ... }` entries over leaf prediction values.

For classification trees, prediction values are the stored class labels of the leaves. For regression trees, they are the stored numeric leaf values.

### `tree_node(...)`, `tree_level(...)`, and `tree_leaf(...)`

These expose the same semantic records that appear in IR export:

- standard trees use `tree_node(...)` to inspect splits, child links, unmatched leaves for multiway nodes, and node stats
- oblivious trees use `tree_level(...)` to inspect one shared split per depth
- all trees use `tree_leaf(...)` to inspect leaf payloads and stats

Returned records include the learned cutoff information and whatever training stats are available there, such as sample counts, impurity, gain, class counts, or variance.

### Examples

Standard tree:

```python
model = train(X, y, task="classification", tree_type="cart")

summary = model.tree_structure()
root = model.tree_node(0)
leaf = model.tree_leaf(0)

print(summary["actual_depth"])
print(root["split"])
print(leaf["leaf"])
```

Oblivious tree:

```python
model = train(X, y, task="classification", tree_type="oblivious")

summary = model.tree_structure()
level0 = model.tree_level(0)
leaf0 = model.tree_leaf(0)

print(summary["representation"])
print(level0["split"])
print(leaf0["stats"])
```

Forest:

```python
forest = train(X, y, algorithm="rf", task="classification", tree_type="cart")

print(forest.tree_count)
print(forest.tree_structure(tree_index=3))
print(forest.tree_node(0, tree_index=3))
print(forest.to_dataframe(tree_index=3).head())
```

### How it works

The introspection API is IR-backed:

- standard trees are exposed as `node_tree`
- oblivious trees are exposed as `oblivious_levels`
- optimized models reuse the same semantic representation as the source model

So introspection reflects the actual learned tree semantics, not a separate debugging-only view.

## Benchmarks

Run:

- `task benchmark-inference`

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

## Serialization and IR

### Model serialization

Use:

- `model.serialize(pretty=False)`
- `Model.deserialize(serialized)`

This round-trips the current model through the JSON IR.

### IR export

Use:

- `model.to_ir_json(pretty=False)`

### Compiled optimized artifacts

Optimized runtimes also support a dedicated compiled binary artifact:

- `compiled = optimized.serialize_compiled()`
- `restored = OptimizedModel.deserialize_compiled(compiled, physical_cores=None)`

This is not a second semantic model format. The split is:

- JSON IR for portability, inspection, and semantic stability
- compiled artifact for faster reload of the optimized CPU runtime

The compiled artifact stores both:

- the semantic model payload needed to preserve the same IR
- the already-lowered optimized runtime layout

So reloading a compiled artifact avoids repeating the runtime-lowering step that `optimize_inference(...)` normally performs.

The IR is designed to be inference-complete for the features implemented today. It records:

- algorithm, task, tree type, and criterion
- explicit tree structure as `node_tree` or `oblivious_levels`
- training-time numeric bin boundaries
- node and leaf stats such as sample counts, impurity, gain, class counts, and variance where applicable

Current IR v1 intentionally marks these as unsupported rather than pretending they exist:

- missing-value handling
- categorical preprocessing semantics

## Current support matrix

- regression: `cart`, `randomized`, `oblivious`
- classification: `id3`, `c45`, `cart`, `randomized`, `oblivious`

## Development

From the repo root:

```bash
task setup-local-env
task python-ext-develop
task test
```
