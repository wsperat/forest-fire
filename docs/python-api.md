# Python API

The Python surface is centered on:

- `Table`
- `train(...)`
- `Model`
- `OptimizedModel`

## Training

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
    max_depth=None,
    min_samples_split=None,
    min_samples_leaf=None,
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

### Supported values

- `algorithm="dt" | "rf" | "gbm"`
- `task="auto" | "regression" | "classification"`
- `tree_type="id3" | "c45" | "cart" | "randomized" | "oblivious"`
- `criterion="auto" | "gini" | "entropy" | "mean" | "median"`

### Parameter semantics

#### `algorithm`

- `dt`: one tree
- `rf`: bagged ensemble with bootstrap sampling and feature subsampling
- `gbm`: stage-wise second-order boosting with shrinkage and gradient-focused row sampling

#### `task`

`task="auto"` infers:

- classification for integer, boolean, and string targets
- regression for float targets

#### `tree_type`

- `id3`: entropy-first classifier
- `c45`: practical extension of ID3
- `cart`: standard binary tree
- `randomized`: stochastic split-search variant
- `oblivious`: symmetric tree with one split per depth

#### `criterion`

Current `auto` behavior:

- `id3`, `c45` classification -> `entropy`
- classification `cart`, `randomized`, `oblivious` -> `gini`
- regression models -> `mean`
- `gbm` trains second-order trees internally when `criterion="auto"`

#### `canaries`

Canaries are shuffled copies of already-preprocessed features used for automatic growth stopping.

Current stopping behavior:

- standard trees stop at the current node
- oblivious trees stop the remaining depth growth
- `gbm` stops adding new stages when the first/root split that would be taken is a canary

#### `bins`

Current values:

- `"auto"`
- integer `1..=512`

Current `auto` behavior:

- per numeric feature, ForestFire picks the highest power of two up to `512`
- each realized bin must contain at least two rows
- the chosen count is capped by the number of distinct observed values

#### `physical_cores`

This controls CPU usage during fitting. ForestFire uses physical cores as the public knob because split scoring is memory-sensitive and that is a better limit than logical threads for this workload.

#### `compute_oob`

Only meaningful for `algorithm="rf"`.

- exposes `model.compute_oob`
- exposes `model.oob_score`
- classification uses OOB accuracy
- regression uses OOB `R^2`

#### `learning_rate`

Only meaningful for `algorithm="gbm"`.

- each stage prediction is multiplied by `learning_rate`
- lower values generally require more trees

#### `bootstrap`

Only meaningful for `algorithm="gbm"`.

- `False`: each stage starts from the full table, then applies gradient-focused row sampling
- `True`: each stage first draws a bootstrap sample, then applies gradient-focused row sampling

#### `top_gradient_fraction` and `other_gradient_fraction`

Only meaningful for `algorithm="gbm"`.

- `top_gradient_fraction` keeps the largest-gradient rows
- `other_gradient_fraction` samples additional rows from the remainder

## Supported input types

- NumPy arrays
- Python sequences
- pandas
- polars
- pyarrow
- SciPy dense matrices
- SciPy sparse matrices

Single-row prediction also accepts 1D inputs like:

- `[1, 2, 3]`
- `np.array([1, 2, 3])`

The key API distinction is:

- training can use raw inputs or an explicit `Table`
- inference should normally use raw inputs directly

That means `Table` is a training-oriented preprocessing container, not the preferred prediction input type.

## Missing values

ForestFire accepts the common missing-value representations that usually appear
through those inputs:

- Python `None`
- floating-point `NaN`
- pandas/NumPy `NaN`
- `polars` null values

Training and prediction treat those values as missing rather than rejecting
them.

The split semantics are:

- each feature reserves a separate missing bin
- split search ignores that bin when choosing the best observed threshold or branch grouping
- once the best observed split has been found, the learner evaluates sending missing rows left vs right and stores the better routing decision
- if a feature had no missing rows at a learned split, a later missing value falls back to the node prediction instead of pretending that the feature had seen a trained missing branch

That fallback is:

- majority class or node probabilities for classification
- node mean prediction for regression

## Tables and input handling

### `Table`

`Table` is the public container for validated training data. You can pass raw data directly to `train(...)`, but building a `Table` explicitly is useful when you want preprocessing and validation separated from fitting.

`Table` chooses between:

- `DenseTable` for mixed numeric/binary data
- `SparseTable` for binary sparse inputs

Why `Table` exists at all:

- training wants one normalized, validated, binned representation
- all learners should see the same preprocessing contract
- canaries, auto binning, and sparse-vs-dense decisions belong in one place

Why `Table` is not the main inference abstraction:

- inference often starts from raw application data
- prediction should not require users to construct a training-oriented container first
- optimized runtimes now do their own lightweight projected preprocessing directly from raw inputs

### `DenseTable`

`DenseTable` is Arrow-backed and optimized for repeated feature scans. Numeric features are rank-binned into a power-of-two number of bins, using the highest populated count up to `512` by default while keeping at least two rows per realized bin, and binary `0/1` columns are stored as booleans.

That combination is deliberate:

- Arrow arrays provide compact columnar storage
- power-of-two bins make later runtime layouts simpler
- forcing at least two rows per realized bin avoids wasting domain size on near-empty bins
- boolean storage keeps true binary columns cheap during both training and inference

Each dense feature also reserves one extra bin for missing values so split
search can handle missingness without rebucketing the data at every node.

### `SparseTable`

`SparseTable` is binary-only. Internally it stores, per feature, the row positions where the value is `1`, so memory usage scales with the positive entries rather than the full dense shape.

This is useful because many sparse inputs are really presence/absence matrices. In that case the right abstraction is not “a giant mostly-zero dense matrix”; it is “which rows contain a positive value for each feature”.

## Main model methods

- `predict(...)`
- `predict_proba(...)`
- `optimize_inference(...)`
- `serialize(...)`
- `to_ir_json(...)`
- `to_dataframe(...)`

## Optimized inference

`optimize_inference(...)` returns an `OptimizedModel` that preserves model semantics while lowering execution into a runtime-oriented representation.

Python signature:

```python
optimized = model.optimize_inference(
    physical_cores=None,
    missing_features=None,
)
```

Key runtime changes:

- CART-style binary trees use compact fallthrough/jump layouts
- multiway classifier splits use dense lookup tables
- oblivious trees use compact level arrays
- optimized models project inputs down to the features that actually appear in splits
- forests and boosted ensembles are lowered in a feature-locality-friendly tree order
- multi-row inputs are preprocessed together before scoring
- compiled binary and oblivious runtimes use compact column-major binned matrices
- row batches are scored in parallel across physical cores

The runtime pipeline is:

1. inspect the semantic model
2. compute `used_feature_indices`
3. lower trees into runtime-friendly structures
4. accept raw inference input
5. validate it against the semantic schema
6. preprocess only the projected feature subset
7. score rows through scalar or batch-oriented execution

That is why optimized inference can reduce total latency even when the tree traversal itself is only moderately faster: it often avoids preprocessing columns that were never going to be used.

### Missing checks in optimized runtimes

By default, optimized runtimes preserve missing-aware inference for every used
feature:

```python
optimized = model.optimize_inference()
```

If you know that only some semantic feature indices may be missing at
prediction time, pass them explicitly:

```python
optimized = model.optimize_inference(missing_features=[0, 4, 9])
```

Semantics:

- `missing_features=None`: keep missing checks for every used feature
- `missing_features=[...]`: only optimized nodes that split on those semantic feature indices keep explicit missing handling
- `missing_features=[]`: omit missing checks entirely in the optimized runtime

This is a runtime-only optimization knob. Use it only when you control
inference inputs and know which columns can actually be missing. Otherwise, the
default is the safe choice. If an excluded feature later arrives missing, the
optimized model will not execute the learned missing-specific branch for that
split.

### Using runtime metadata

The most useful runtime inspection values are:

- `model.used_feature_indices`
- `model.used_feature_count`
- `optimized.used_feature_indices`
- `optimized.used_feature_count`

Example:

```python
optimized = model.optimize_inference()

print(model.used_feature_indices)
print(optimized.used_feature_count)
```

Those values are semantic, not profiler-derived. They come from the trained splits in the model.

It helps most on:

- large prediction batches
- deeper trees
- repeated scoring of the same model
- compiled binary trees
- wide inputs where the trained model only touches a small subset of columns

### Compiled optimized artifacts

An optimized Python model can also be serialized into a compiled artifact:

```python
optimized = model.optimize_inference()
payload = optimized.serialize_compiled()
restored = forestfire.OptimizedModel.deserialize_compiled(payload)
```

That artifact contains:

- the semantic IR
- the lowered runtime layout
- the feature projection metadata

This is useful when you want:

- faster reloads of the optimized runtime
- the same optimized execution strategy after deserialization
- one deployment artifact for repeated scoring

The semantic JSON serialization and the compiled optimized artifact solve different problems. The JSON form is the canonical model meaning; the compiled artifact is the cached execution form.

## Introspection

- `tree_count`
- `tree_structure(...)`
- `tree_prediction_stats(...)`
- `tree_node(...)`
- `tree_level(...)`
- `tree_leaf(...)`

`to_dataframe(...)` returns a `polars.DataFrame` when `polars` is installed and falls back to a `pyarrow.Table` otherwise.

Typical use cases:

- understanding realized tree shape after training
- inspecting cutoffs and leaf payloads
- summarizing leaf prediction distributions
- inspecting one tree at a time inside a forest or boosted ensemble
