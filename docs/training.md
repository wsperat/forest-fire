# Training

## Algorithms

- `dt`: one tree
- `rf`: random forest
- `gbm`: gradient boosting

These are separate algorithms, not just cosmetic wrappers around the same tree builder:

- `dt` optimizes a single tree directly and is useful when interpretability matters most
- `rf` reduces variance by averaging many independently sampled trees
- `gbm` fits trees stage-by-stage to residual structure and is meant to trade interpretability for stronger predictive performance

## Support matrix

- `algorithm="dt"`
  - regression: `cart`, `randomized`, `oblivious`
  - classification: `id3`, `c45`, `cart`, `randomized`, `oblivious`
- `algorithm="rf"`
  - regression: `cart`, `randomized`, `oblivious`
  - classification: `id3`, `c45`, `cart`, `randomized`, `oblivious`
- `algorithm="gbm"`
  - regression: `cart`, `randomized`, `oblivious`
  - classification: `cart`, `randomized`, `oblivious` with binary targets only

The support matrix is narrower for boosting because boosting needs more than a split criterion and a leaf value. It needs consistent gradient/Hessian handling, additive stage updates, and stable leaf semantics under repeated residual fitting. That is why the current boosting support is focused on the binary-split regression-style tree families that map cleanly onto second-order stage training.

## Tree types

Classification:

- `id3`
- `c45`
- `cart`
- `randomized`
- `oblivious`

Regression:

- `cart`
- `randomized`
- `oblivious`

Why multiple tree types exist:

- `id3` and `c45` keep the classic multiway classifier style and are useful when you want a direct, explicit branch-per-bin structure
- `cart` is the general-purpose binary-tree baseline and the most natural fit for forests and boosting
- `randomized` introduces stochasticity into split search, which is useful both as a standalone learner and as the basis for extra-trees-style ensembles
- `oblivious` enforces one split per depth across the tree, which makes the final model more regular and easier to lower into compact runtime layouts

## Task detection

With `task="auto"`:

- integer, boolean, and string targets become classification
- float targets become regression

This keeps the common path ergonomic without hiding the important distinction that task choice changes:

- the split objective
- the leaf payload
- the valid tree families
- the prediction API surface, especially `predict_proba(...)`

## Missing values during training

ForestFire treats missing values as first-class training inputs rather than as
rows that must be dropped beforehand.

Accepted missing markers include:

- Python `None`
- floating-point `NaN`
- pandas/NumPy `NaN`
- `polars` null values

The training contract is:

- every feature gets a dedicated missing bin
- observed split search ignores that missing bin
- missing rows are then routed according to the configured
  `missing_value_strategy`

The public strategies are:

- `"heuristic"`: choose the best split from observed values first, then decide whether missing rows should go left or right for that split
- `"optimal"`: for every candidate split, evaluate missing-left and missing-right, then choose the best full combination
- per-column dictionary: assign `"heuristic"` or `"optimal"` feature by feature, with unspecified features defaulting to `"heuristic"`

Python examples:

```python
train(X, y, missing_value_strategy="heuristic")
train(X, y, missing_value_strategy="optimal")
train(X, y, missing_value_strategy={"col_1": "optimal", "col_2": "heuristic"})
train(X, y, missing_value_strategy={"f0": "optimal", "f1": "heuristic"})
```

The feature-name forms are aliases for semantic column indices:

- `"col_1"` and `"f0"` both mean feature index `0`
- `"col_2"` and `"f1"` both mean feature index `1`

Why both strategies exist:

- `"heuristic"` is the practical default and keeps training cost closer to ordinary split search
- `"optimal"` can be much slower because it expands the search to all candidate split and missing-routing combinations

That design matters because it separates two questions that are often muddled
together:

- what is the best split among observed values?
- given that split, where should the missing rows go?

When a node never saw missing values for its split feature during training, the
model does not invent a learned missing branch. A later missing value at
prediction time falls back to the node prediction instead:

- majority class for classification
- node mean for regression

Current implementation note:

- the strategy toggle is implemented for the standard first-order tree builders
- the second-order boosting path currently uses the existing missing-value behavior rather than a separate heuristic-vs-optimal toggle

## Stopping and control parameters

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `seed`

These parameters materially affect learning across the supported tree families, including forest constituent trees.

The rationale behind these controls is straightforward:

- `max_depth` limits structural complexity directly and is the cleanest way to cap traversal cost
- `min_samples_split` prevents very small unstable internal nodes
- `min_samples_leaf` prevents highly confident leaves built from tiny support
- `max_features` controls correlation and search width, which matters much more in ensembles than in a single tree
- `seed` is there because the library deliberately uses randomness in several places: bootstrapping, feature subsampling, randomized splits, and boosting row sampling

The intended semantics of `seed` are:

- fixed seed -> repeatable training for the same dataset and configuration
- different seed -> legitimately different stochastic choices

That randomness is not left ad hoc inside each learner. ForestFire now uses a
shared seed-mixing path for:

- forest tree seeds
- boosting stage seeds
- node-local feature-subset sampling
- randomized threshold selection

One practical detail matters here: node-local randomization is derived from the
rows owned by the node, but not from the incidental order of the mutable
row-index buffer. That makes the stochastic behavior more stable under internal
partitioning details while remaining sensitive to the actual training data seen
at each node.

## Canaries

ForestFire uses canary features to stop growth during training.

For the full rationale and algorithm-specific behavior, see [Canary Strategy](canaries.md).

Behavior:

- standard trees stop at the current node
- oblivious trees stop further depth growth
- gradient boosting stops adding stages if the next stage’s root split would be a canary
- random forests ignore canaries during tree training

Why canaries exist:

- standard tree learners can always continue to fit noise if the stopping rule only looks at local impurity decrease
- a canary feature is a shuffled copy of real signal space, so if the learner prefers it, the current split quality is no better than structured noise
- this turns stopping into a training-time competition between real and noise-like features instead of a later pruning pass

Why random forests ignore them:

- forests already reduce overfitting primarily through bootstrapping and feature subsampling
- reintroducing canaries inside each tree would often stop growth for the wrong reason, because each constituent tree already sees a deliberately restricted view of the data
- in boosting, the opposite tradeoff is useful: canaries remain active because stage-wise residual fitting is much more willing to chase noise late in training

## Random forests

- bootstrap sampling per tree
- feature subsampling per node
- optional OOB score computation

The random-forest implementation is intentionally close to the classical idea:

- each tree trains on a bootstrap sample
- each node considers only a sampled feature subset
- predictions are aggregated by averaging leaf outputs or probabilities

The important design choice is that preprocessing is still shared. Trees do not each rebucket the raw data. They all operate over the same preprocessed representation, which keeps training semantics aligned and avoids duplicated preprocessing work.

## Gradient boosting

- second-order trees trained from gradients and Hessians
- `learning_rate`
- optional stage bootstrapping
- LightGBM-style gradient-focused row sampling via:
  - `top_gradient_fraction`
  - `other_gradient_fraction`

The boosting implementation is LightGBM-like in spirit, but not a clone.

It keeps the ideas that matter most:

- second-order tree fitting from gradients and Hessians
- shrinkage via `learning_rate`
- optional row sampling
- focusing stage fitting on high-gradient rows

But it keeps a ForestFire-specific stopping rule:

- canaries still participate in split selection
- if the root split of the next stage would be a canary, that stage is discarded and boosting stops

That choice reflects the project’s general design preference: use explicit training-time noise competition as a stopping signal instead of bolting on a separate pruning or early-stopping layer later.

## Training optimizations

ForestFire training is optimized around a compact binned core and shared row-index buffers rather than row copies.

- numeric features are pre-binned into compact integer ranks, capped at `512` bins
- each feature reserves one extra missing bin alongside the observed bins
- `bins="auto"` chooses the highest populated power-of-two count per feature while keeping at least two rows in every realized bin
- `histogram_bins=...` can override the numeric resolution used during split search without requiring callers to rebuild the source `Table`
- long-running training and prediction release the Python GIL before entering the Rust hot path
- CART and randomized trees use histogram-based numeric split search
- standard binary trees partition row indices in place
- ID3 and C4.5 use the same in-place row-buffer approach for multiway branches
- oblivious trees train over shared row buffers plus per-leaf ranges
- CART/randomized classification and mean-regression builders reuse parent histograms and derive sibling histograms by subtraction
- oblivious split scoring reuses cached per-leaf counts or `sum`/`sum_sq`
- random forests parallelize across trees while limiting intra-tree parallelism
- binary sparse inputs stay sparse through training and inference
- classifier, regressor, and second-order tree builders now share the same core
  histogram/partitioning/randomization helpers instead of carrying separate
  implementations of the same mechanics

### Reading the snippets below

As in the runtime page, the snippets here are representative shapes rather than exact copies of every production function.

They are meant to show:

- the straightforward naïve way to structure the work
- the shape ForestFire actually optimizes for
- why the optimized shape is cheaper in memory traffic, allocation, or repeated computation

The real implementations live across `crates/data/src` and `crates/core/src/tree`, but these snippets are the shortest path to understanding the architecture.

### 1. Binning once instead of rescanning raw floats at every node

#### Naïve approach

```rust
fn best_threshold_naive(values: &[f64], rows: &[usize]) -> Option<f64> {
    let mut best = None;
    for &candidate_row in rows {
        let threshold = values[candidate_row];
        let (left, right): (Vec<_>, Vec<_>) = rows
            .iter()
            .copied()
            .partition(|&row| values[row] <= threshold);
        let score = score_split_from_raw_values(values, &left, &right);
        best = pick_better(best, (threshold, score));
    }
    best.map(|(threshold, _)| threshold)
}
```

This rescans raw floating-point values and repartitions rows for every candidate threshold.

#### ForestFire approach

```rust
fn best_threshold_binned(
    binned_values: &[u16],
    histogram: &[HistogramBin],
) -> Option<u16> {
    let mut best = None;
    let mut left = HistogramBin::default();
    let total = histogram_total(histogram);

    for (bin, bucket) in histogram.iter().enumerate() {
        left = left + bucket;
        let right = total - left;
        let score = score_split_from_histograms(&left, &right);
        best = pick_better(best, (bin as u16, score));
    }

    best.map(|(bin, _)| bin)
}
```

Why this helps:

- candidate space is bounded by the realized bins
- split scoring works from aggregated counts or sums rather than rescanning rows
- the same binned representation is reused by every learner family and later by optimized inference

### 2. In-place row partitioning instead of copying child row vectors

#### Naïve approach

```rust
fn split_rows_naive(rows: &[usize], feature: usize, threshold: u16, table: &dyn TableAccess) -> (Vec<usize>, Vec<usize>) {
    rows.iter()
        .copied()
        .partition(|row| table.binned_value(feature, *row) <= threshold)
}
```

This allocates new child vectors at every split.

#### ForestFire approach

```rust
fn split_rows_in_place(
    rows: &mut [usize],
    feature: usize,
    threshold: u16,
    table: &dyn TableAccess,
) -> usize {
    let mut boundary = 0usize;
    for idx in 0..rows.len() {
        if table.binned_value(feature, rows[idx]) <= threshold {
            rows.swap(boundary, idx);
            boundary += 1;
        }
    }
    boundary
}
```

Now one mutable row-index buffer is reused throughout growth:

- `rows[..boundary]` is the left child
- `rows[boundary..]` is the right child

Why this helps:

- far fewer allocations
- better cache locality on the row-index buffer
- easier reuse of the same data structure across CART, randomized, and second-order trees

### 3. Histogram subtraction instead of rebuilding sibling statistics

#### Naïve approach

```rust
fn child_histograms_naive(
    left_rows: &[usize],
    right_rows: &[usize],
    table: &dyn TableAccess,
) -> (Vec<HistogramBin>, Vec<HistogramBin>) {
    (
        build_histograms_from_rows(left_rows, table),
        build_histograms_from_rows(right_rows, table),
    )
}
```

This does full work for both children even though the parent histogram is already known.

#### ForestFire approach

```rust
fn child_histograms_subtractive(
    parent: &[HistogramBin],
    left: &[HistogramBin],
) -> (Vec<HistogramBin>, Vec<HistogramBin>) {
    let right = parent
        .iter()
        .zip(left.iter())
        .map(|(parent_bin, left_bin)| parent_bin.subtract(left_bin))
        .collect::<Vec<_>>();
    (left.to_vec(), right)
}
```

Why this helps:

- one child histogram can be built directly
- the sibling becomes derived work instead of independent work
- the savings compound at every internal node in binary tree growth

### 4. Shared row buffers for oblivious training

#### Naïve approach

```rust
fn grow_oblivious_naive(leaves: &[Vec<usize>], split: Split, table: &dyn TableAccess) -> Vec<Vec<usize>> {
    let mut next = Vec::new();
    for leaf_rows in leaves {
        let (left, right) = leaf_rows
            .iter()
            .copied()
            .partition(|row| table.binned_value(split.feature_index, *row) <= split.threshold_bin);
        next.push(left);
        next.push(right);
    }
    next
}
```

This keeps allocating fresh vectors per leaf at every depth.

#### ForestFire approach

```rust
struct LeafRange {
    start: usize,
    end: usize,
}

fn grow_oblivious_in_place(
    rows: &mut [usize],
    leaves: &[LeafRange],
    split: Split,
    table: &dyn TableAccess,
) -> Vec<LeafRange> {
    let mut next = Vec::with_capacity(leaves.len() * 2);
    for leaf in leaves {
        let boundary = split_rows_in_place(
            &mut rows[leaf.start..leaf.end],
            split.feature_index,
            split.threshold_bin,
            table,
        );
        next.push(LeafRange {
            start: leaf.start,
            end: leaf.start + boundary,
        });
        next.push(LeafRange {
            start: leaf.start + boundary,
            end: leaf.end,
        });
    }
    next
}
```

Why this helps:

- one row buffer is reused for the whole tree
- leaves are tracked by ranges instead of fresh row vectors
- this matches the symmetric structure of oblivious trees and keeps the implementation regular

### 5. Deterministic randomization through shared seed mixing

#### Naïve approach

```rust
let mut rng = StdRng::seed_from_u64(user_seed);
let threshold = candidates[rng.gen_range(0..candidates.len())];
```

This makes results depend on the accidental order in which random draws happen across nodes, trees, or stages.

#### ForestFire approach

```rust
let tree_seed = mix_seed(base_seed, tree_index as u64);
let node_seed = node_seed(tree_seed, depth, salt, rows);
let threshold = choose_random_threshold(&candidates, feature_index, rows, node_seed)?;
```

Why this helps:

- fixed seed means repeatable training
- different trees, stages, and nodes still get different random contexts
- node-local choices depend on the row set, not on incidental temporary ordering

That is important for randomized trees, forests, and gradient boosting stages alike.

### 6. Parallelize across trees where independence is real

#### Naïve approach

```rust
let trees = (0..n_trees)
    .map(|tree_index| train_one_tree(sample_for(tree_index)))
    .collect::<Result<Vec<_>, _>>()?;
```

This is correct, but it leaves independent bootstrap trees serialized.

#### ForestFire approach

```rust
let trees = (0..n_trees)
    .into_par_iter()
    .map(|tree_index| train_one_tree(sample_for(tree_index)))
    .collect::<Result<Vec<_>, _>>()?;
```

Why this helps:

- forest trees are naturally independent training jobs
- the speedup lands at the algorithm level without changing single-tree semantics
- intra-tree work can stay conservative while inter-tree work scales out

ForestFire deliberately prefers this shape for forests because the independence is obvious and the coordination cost is low.

### 7. Stage-wise boosting on shared preprocessed data

#### Naïve approach

```rust
for stage in 0..n_trees {
    let residual_table = rebuild_table_from_current_residuals(x, residuals);
    let tree = train_tree(&residual_table)?;
    update_predictions(&mut predictions, &tree);
}
```

This rebuilds too much state per stage.

#### ForestFire approach

```rust
for stage in 0..n_trees {
    let (gradients, hessians) = compute_gradients_and_hessians(&predictions, targets);
    let sampled_rows = gradient_focus_sample(base_rows, &gradients, &hessians, ...);
    let sampled_table = SampledTable::new(train_set, sampled_rows.row_indices);
    let tree = train_second_order_tree(&sampled_table, &sampled_rows.gradients, &sampled_rows.hessians)?;
    update_predictions(&mut predictions, &tree, learning_rate);
}
```

Why this helps:

- the same preprocessed base table is reused
- only gradients, Hessians, and sampled row views change per stage
- stage computation stays focused on residual structure instead of rebuilding the whole world

### 8. Keep sparse binary inputs sparse

#### Naïve approach

```rust
fn sparse_to_dense(x: SparseInput) -> Vec<Vec<f64>> {
    let mut dense = vec![vec![0.0; x.n_features]; x.n_rows];
    for (feature, rows) in x.nonzero_rows {
        for row in rows {
            dense[row][feature] = 1.0;
        }
    }
    dense
}
```

This pays the full dense memory cost before training even starts.

#### ForestFire approach

```rust
struct SparseBinaryColumn {
    row_indices: Vec<usize>,
}

struct SparseTable {
    columns: Vec<SparseBinaryColumn>,
    n_rows: usize,
}
```

Why this helps:

- storage scales with positive entries, not full shape
- binary-feature semantics stay explicit
- the same sparse structure can feed both training and inference paths

### 9. Share core mechanics across learner families

#### Naïve approach

```rust
mod classifier_training {
    fn build_histograms(...) { ... }
    fn partition_rows(...) { ... }
}

mod regressor_training {
    fn build_histograms(...) { ... }
    fn partition_rows(...) { ... }
}

mod second_order_training {
    fn build_histograms(...) { ... }
    fn partition_rows(...) { ... }
}
```

This invites drift:

- one path gets faster
- another keeps the old bug or old allocation pattern
- randomization semantics stop lining up

#### ForestFire approach

```rust
mod tree::shared {
    fn candidate_feature_indices(...) { ... }
    fn partition_rows_for_binary_split(...) { ... }
    fn choose_random_threshold(...) { ... }
    fn mix_seed(...) { ... }
}
```

Why this helps:

- performance-sensitive fixes land once
- first-order and second-order trees stay behaviorally aligned
- the codebase can evolve without silently splitting into several incompatible training engines

Why these optimizations matter:

- binning turns repeated threshold search into a bounded discrete problem, which makes both scoring and runtime lowering much cheaper
- shared row-index buffers avoid per-node row-vector allocation, which is one of the main hidden costs in naive tree builders
- histogram subtraction matters because sibling statistics are not independent work; once one child is known, the other is often derivable from the parent
- shared tree internals matter because performance-sensitive fixes to histogram
  handling, partitioning, and stochastic split selection now land once and
  propagate across first-order and second-order learners together
- keeping sparse binary inputs sparse avoids paying a dense-memory penalty for data that is structurally sparse

Impact in practice:

- better cache behavior during split search
- less allocator pressure during recursive growth
- more predictable runtime layouts for optimized inference
- much better scaling in forests, where repeated tree training amplifies every avoidable allocation and recounting cost
