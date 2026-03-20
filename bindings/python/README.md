# forestfire (Python)

Python bindings for the unified ForestFire training interface.

The Python package is built around five objects:

- `Table` for validated training data
- `train(...)` for fitting
- `Model.predict(...)` for inference
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
    physical_cores=4,
)

preds = model.predict(X)
fast_model = model.optimize_inference(physical_cores=4)
fast_preds = fast_model.predict(X)
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
    task="regression",
    tree_type="target_mean",
    criterion="auto",
    canaries=2,
    physical_cores=None,
)
```

### Current supported values

- `algorithm="dt"`
- `task="regression" | "classification"`
- `tree_type="target_mean" | "id3" | "c45" | "cart" | "oblivious"`
- `criterion="auto" | "gini" | "entropy" | "mean" | "median"`

### Why these parameters exist

#### `algorithm`

Only `dt` exists today, but the parameter is already part of the public surface so new learner families can be added without breaking the top-level API.

#### `task`

ForestFire does not guess whether `y` means regression or classification. The task is explicit because it changes split scoring, leaf semantics, defaults, and the set of valid tree types.

#### `tree_type`

`tree_type` selects the structural family directly:

- `target_mean`: regression baseline
- `id3`: entropy-first classifier
- `c45`: practical extension of ID3
- `cart`: standard binary tree
- `oblivious`: symmetric tree with one split per depth

#### `criterion`

Criterion changes the model itself, not just training speed:

- `gini` and `entropy` for classification
- `mean` and `median` for regression
- `auto` to pick the default implied by task and tree type

Current `auto` behavior:

- `id3`, `c45` -> `entropy`
- classification `cart`, `oblivious` -> `gini`
- regression models -> `mean`

#### `canaries`

ForestFire uses automatic growth stopping instead of pruning. Canary variables are shuffled copies of the already-preprocessed features. If a learner chooses a canary, it has reached a noise-like split and stops growing there.

Current stopping behavior:

- standard trees stop at the current node
- oblivious trees stop the remaining depth growth

#### `physical_cores`

This controls CPU usage during fitting. The library uses physical cores as the public knob because split scoring is memory-sensitive and that tends to be a more honest resource limit than logical threads.

## Tables and input handling

### `Table`

`Table` is the public container for validated training data. You can pass raw data directly to `train(...)`, but building a `Table` explicitly is useful when you want preprocessing and validation separated from fitting.

`Table` is intentionally training-oriented. For inference, the intended path is to pass raw arrays, dicts, dataframes, lazyframes, or sparse matrices directly to `predict(...)`.

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

`DenseTable` is Arrow-backed and optimized for repeated feature scans. Numeric features are rank-binned into `512` bins, and binary `0/1` columns are stored as booleans so they are both smaller and cheaper to split on.

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
- compiled binary and oblivious runtimes use column-major binned matrices so one split can scan many rows at once
- `polars.LazyFrame` inputs are collected and scored in batches of about `10_000` rows
- row batches are scored in parallel across the requested physical cores

### What that means at the CPU level

#### Prediction-only node layouts

The optimized runtime removes training-only fields from the hot prediction path. That reduces object size, cache pressure, and the amount of general-case logic the predictor loop has to carry around.

#### Compiled CART-style fallthrough layout

For binary trees, the optimized runtime keeps the more common child as the next node in memory and stores only the less common branch as an explicit jump target. That shrinks the hot node representation and reduces branch-heavy traversal logic.

#### Dense lookup for multiway nodes

For multiway classifier nodes, the optimized runtime replaces “scan branches until one matches” with “index the precomputed child table by bin id”. That reduces dependent comparisons and makes access more regular.

#### Compact oblivious-tree loops

Oblivious trees become a sequence of feature-index and threshold arrays plus a final leaf array. Scoring becomes a fixed loop that accumulates a leaf index, which is much more regular than standard pointer-chasing tree traversal.

#### Whole-batch preprocessing and column-major batches

Inference inputs are converted into compact bin ids before the optimized traversal runs. For compiled binary and oblivious runtimes, those bins are arranged column-major so the predictor can read one feature across many rows before moving on.

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
- `target_mean`
- very shallow trees
- workloads dominated by input conversion instead of traversal
- some single-core oblivious workloads, where layout conversion can still dominate

### Why the IR stays the same

`OptimizedModel` delegates serialization and IR export to the original semantic model. That keeps portability stable: optimization changes execution strategy, not model meaning.

## Serialization and IR

### Model serialization

Use:

- `model.serialize(pretty=False)`
- `Model.deserialize(serialized)`

This round-trips the current model through the JSON IR.

### IR export

Use:

- `model.to_ir_json(pretty=False)`

The IR is designed to be inference-complete for the features implemented today. It records:

- algorithm, task, tree type, and criterion
- explicit tree structure as `node_tree` or `oblivious_levels`
- training-time numeric bin boundaries
- node and leaf stats such as sample counts, impurity, gain, class counts, and variance where applicable

Current IR v1 intentionally marks these as unsupported rather than pretending they exist:

- missing-value handling
- categorical preprocessing semantics

## Current support matrix

- regression: `target_mean`, `cart`, `oblivious`
- classification: `id3`, `c45`, `cart`, `oblivious`

## Development

From the repo root:

```bash
task setup-local-env
task python-ext-develop
task test
```
