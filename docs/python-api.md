# Python API

The Python surface is centered on:

- `Table`
- `train(...)`
- `Model`
- `OptimizedModel`
- sklearn-compatible wrappers in `forestfire.tree`, `forestfire.forest`, and `forestfire.gbm`

## Training

```python
train(
    x,
    y=None,
    algorithm="dt",
    task="auto",
    tree_type="cart",
    split_strategy="axis_aligned",
    builder="greedy",
    lookahead_depth=1,
    lookahead_top_k=8,
    lookahead_weight=1.0,
    beam_width=4,
    criterion="auto",
    canaries=2,
    bins="auto",
    histogram_bins=None,
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
    missing_value_strategy=None,
    categorical_strategy=None,
    categorical_features=None,
    target_smoothing=20.0,
    filter=None,
)
```

### Supported values

- `algorithm="dt" | "rf" | "gbm"`
- `task="auto" | "regression" | "classification"`
- `tree_type="id3" | "c45" | "cart" | "randomized" | "oblivious"`
- `split_strategy="axis_aligned" | "oblique"`
- `builder="greedy" | "lookahead" | "beam"`
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

#### `split_strategy`

- `axis_aligned`: ordinary one-feature threshold splits
- `oblique`: two-feature linear splits of the form `w1 * x_i + w2 * x_j <= t`

Current support matrix:

- `axis_aligned`: supported everywhere
- `oblique`: supported for `dt`, `rf`, and `gbm` when `tree_type` is `cart` or
  `randomized`

Current oblique behavior:

- all candidate feature pairs available at the node are considered
- the learned split is still sparse and pairwise: exactly two features per node
- missing values are routed independently per participating feature rather than
  forcing a single node-level missing fallback

#### `criterion`

Current `auto` behavior:

- `id3`, `c45` classification -> `entropy`
- classification `cart`, `randomized`, `oblivious` -> `gini`
- regression models -> `mean`
- `gbm` trains second-order trees internally when `criterion="auto"`

#### `builder`

- `greedy`: ordinary immediate-gain split ranking
- `lookahead`: re-rank the top immediate candidates by one-best-continuation
  future score
- `beam`: re-rank the top immediate candidates by width-limited continuation
  search

Builder controls tree construction strategy independently of:

- `algorithm`
- `tree_type`
- `split_strategy`

Related parameters:

- `lookahead_depth`
- `lookahead_top_k`
- `lookahead_weight`
- `beam_width`

For the detailed behavior, see:

- [Lookahead Builder](lookahead-builder.md)
- [Beam Builder](beam-builder.md)

#### `canaries`

Canaries are shuffled copies of already-preprocessed features used for automatic growth stopping.

Current stopping behavior:

- standard trees stop at the current node when no acceptable real split survives canary competition
- oblivious trees stop the remaining depth growth when no acceptable real level-split survives canary competition
- `gbm` stops adding new stages when no acceptable real root split survives canary competition

Canaries are active for `dt` and `gbm`.

Random forests are the exception:

- `rf` deliberately ignores canaries during tree training
- so `canaries` and `filter` do not affect random-forest growth policy

#### `filter`

`filter` controls how strict canary competition is once split candidates have been scored and ranked.

Accepted forms:

- `None`
- positive integer
- float in `[0, 1)`

The ranking rule is:

1. score split candidates as usual
2. sort them from best to worst
3. look only inside the allowed top window
4. choose the best real feature inside that window
5. if the window contains only canaries, stop growth under the usual canary rule

`filter=None` is the default strict policy and is equivalent to `filter=1`:

- only the single best-ranked candidate is eligible
- if that candidate is a canary, the node stops

If `filter` is an integer `n`:

- the chosen real feature must appear within the top `n` scored candidates
- canaries are still allowed to occupy earlier ranks
- the selected split is the highest-ranked real split inside that top-`n` window

Example:

- `filter=3` means “after sorting all candidates, ignore canaries if needed, but only within the top 3 ranked candidates”

If `filter` is a float `alpha` in `[0, 1)`:

- ForestFire converts it into a top-window fraction of `1 - alpha`
- if there are `k` scored candidates, the allowed window size is `ceil((1 - alpha) * k)`
- the chosen split is again the highest-ranked real split inside that window

Example:

- `filter=0.95` means “look only at the top 5% of ranked candidates”

A few practical details matter:

- the window is computed over scored split candidates, not just over raw input columns
- the exact candidate count can vary by algorithm and by node because `max_features`, tree type, and node-local feasibility all affect how many candidates are actually scorable
- for oblivious trees, the competition happens at the next shared level-split
- for `gbm`, the same logic is applied at the root of the next stage, and if no real split survives inside the allowed window, that whole stage is discarded and boosting stops

#### `bins`

Current values:

- `"auto"`
- integer `1..=512`

Current `auto` behavior:

- per numeric feature, ForestFire picks the highest power of two up to `512`
- each realized bin must contain at least two rows
- the chosen count is capped by the number of distinct observed values

`bins` applies when ForestFire is preprocessing raw training data into a
`Table`. It controls the stored numeric representation of that table.

#### `histogram_bins`

Current values:

- `None`
- `"auto"`
- integer `1..=128`

Semantics:

- `None`: reuse the numeric bins already present in the input training table
- `"auto"` or an integer: rebuild the numeric training view at that resolution
  before split search

This is the estimator-facing control for histogram width. It is separate from
`bins`:

- `bins` controls how a raw Python input is preprocessed into a training table
- `histogram_bins` controls the numeric resolution used by split-search histograms

That distinction matters when:

- you pass an already-built `Table` to `train(...)`
- you want one stored table representation but a different histogram resolution
  during fitting

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

#### `missing_value_strategy`

Controls how split search handles features with missing values.

Accepted forms:

- `"heuristic"`
- `"optimal"`
- `{"col_1": "heuristic", "col_2": "optimal", ...}`
- `{"f0": "heuristic", "f1": "optimal", ...}`

Semantics:

- `"heuristic"`: choose the best split using only observed values first, then evaluate whether the missing rows should go left or right for that chosen split
- `"optimal"`: evaluate missing-left vs missing-right while scoring every candidate split, then choose the overall best combination of split plus missing routing
- dictionary form: apply the chosen strategy per feature, defaulting unspecified features to `"heuristic"`

The dictionary keys use semantic feature indices:

- `"col_1"` means feature index `0`
- `"col_2"` means feature index `1`
- `"f0"` means feature index `0`
- `"f1"` means feature index `1`

Tradeoff:

- `"heuristic"` is much faster and is the default
- `"optimal"` can be substantially slower because it expands the split search

Current implementation note:

- the strategy setting is implemented for the standard first-order tree training paths
- the second-order boosting path uses the same learned missing-routing semantics,
  but it does not expose a separate heuristic-vs-optimal toggle

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

## Sklearn wrappers

ForestFire also exposes sklearn-compatible estimators on top of the Rust
backend.

Import paths:

- `from forestfire.tree import ...`
- `from forestfire.forest import ...`
- `from forestfire.gbm import ...`

Examples:

```python
from forestfire.tree import ObliviousRegressor
from forestfire.forest import CARTRandomForestRegressor
from forestfire.gbm import ExtraGBMRegressor

tree = ObliviousRegressor(max_depth=4).fit(X, y)
forest = CARTRandomForestRegressor(n_estimators=200).fit(X, y)
gbm = ExtraGBMRegressor(n_estimators=100, learning_rate=0.05).fit(X, y)
```

Available wrappers:

- `forestfire.tree`
- `ID3Classifier`
- `C45Classifier`
- `CARTClassifier`
- `ExtraTreeClassifier`
- `ObliviousTreeClassifier`
- `CARTRegressor`
- `ExtraTreeRegressor`
- `ObliviousTreeRegressor`

- `forestfire.forest`
- `ID3RandomForestClassifier`
- `C45RandomForestClassifier`
- `CARTRandomForestClassifier`
- `ExtraRandomForestClassifier`
- `ObliviousRandomForestClassifier`
- `CARTRandomForestRegressor`
- `ExtraRandomForestRegressor`
- `ObliviousRandomForestRegressor`

- `forestfire.gbm`
- `CARTGBMClassifier`
- `ExtraGBMClassifier`
- `ObliviousGBMClassifier`
- `CARTGBMRegressor`
- `ExtraGBMRegressor`
- `ObliviousGBMRegressor`

Sklearn wrapper semantics:

- they call the same Rust-backed `train(...)` API under the hood
- `fit(...)`, `predict(...)`, and classifier `predict_proba(...)` are supported
- fitted estimators expose `model_`
- classifiers expose `classes_`
- fitted estimators expose `n_features_in_` when the input shape is available
- `get_params(...)` and `set_params(...)` are supported
- `sample_weight` is currently rejected

Wrapper defaults intentionally differ from raw `train(...)` in one place:

- sklearn wrappers default to `canaries=0`

That avoids canary-based early stopping surprising users on small sklearn-style
toy datasets.

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
- split search ignores that bin when choosing observed thresholds or branch groupings
- the exact missing-row search behavior then depends on `missing_value_strategy`
- if a feature had no missing rows at a learned split, a later missing value falls back to the node prediction instead of pretending that the feature had seen a trained missing branch

Under `missing_value_strategy="heuristic"`:

- choose the split from observed values first
- then decide whether missing rows should go left or right for that chosen split

Under `missing_value_strategy="optimal"`:

- for each candidate split, evaluate both missing-left and missing-right
- keep the best joint combination of split and missing routing

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
