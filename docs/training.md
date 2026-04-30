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

## Split strategies

ForestFire separates the tree family from the split family.

The public split strategies are:

- `axis_aligned`
- `oblique`

`axis_aligned` is the default and means ordinary one-feature threshold splits.

`oblique` means a sparse pairwise linear split:

```text
w1 * x_i + w2 * x_j <= t
```

Current support matrix:

- `axis_aligned`: all supported tree families
- `oblique`: `dt`, `rf`, and `gbm` when `tree_type` is `cart` or `randomized`

Not currently supported for:

- `id3`
- `c45`
- `oblivious`

Current implementation details:

- oblique nodes are currently pairwise, not arbitrary `k`-feature hyperplanes
- all candidate feature pairs available at a node are considered
- axis-aligned and oblique candidates compete inside the same canary-filtered ranking

## Builders

ForestFire also separates tree family from tree-construction strategy.

The public builders are:

- `greedy`
- `lookahead`
- `beam`
- `optimal`

At a high level:

- `greedy` ranks candidates by immediate node-local score only
- `lookahead` uses a bounded version of the same recursive subtree scorer as
  `optimal`, limited by `lookahead_depth`, `lookahead_top_k`, and
  `lookahead_weight`
- `beam` uses that same bounded scorer, but keeps up to `beam_width`
  continuations alive during future search before taking the strongest
  surviving continuation value
- `optimal` recursively evaluates all legal cuts until ordinary stopping rules
  or canary blocking end that branch

The builder controls how a node decides which split to take. It does not change:

- the split family itself
- the missing-value semantics of the learner
- the leaf payload semantics

Related parameters:

- `lookahead_depth`
- `lookahead_top_k`
- `lookahead_weight`
- `beam_width`

Those tuning knobs apply to `lookahead` and `beam`. `optimal` ignores them.

For the detailed algorithmic behavior, see:

- [Lookahead Builder](lookahead-builder.md)
- [Beam Builder](beam-builder.md)
- [Optimal Builder](optimal-builder.md)

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
- the second-order boosting path uses the same learned missing-routing semantics,
  but it does not expose a separate heuristic-vs-optimal toggle

## Categorical features

Categorical handling uses the same public `train(...)` interface as numeric
training. There is no separate categorical training API.

The current strategies are:

- `dummy`
- `target`
- `fisher`

At a high level:

- `dummy` expands nominal categories into indicator-style derived columns
- `target` replaces categories with smoothed target-derived statistics
- `fisher` orders categories by target-derived statistics and then lets the
  tree learner split over that learned ordering

The implementation is split intentionally:

- `dummy` and `target` are table-layer transforms
- `fisher` is implemented in core because it is tied to category ordering for
  threshold-based split search

From the trainer’s point of view, categorical handling happens before tree
growth and produces the numeric/binary representation consumed by the existing
split machinery. That means:

- trees, forests, and GBM all use the same categorical entry path
- canaries compete against the transformed features rather than raw category
  labels
- histogram-based split search works over the transformed representation

Example:

```python
train(
    X,
    y,
    algorithm="gbm",
    task="classification",
    tree_type="cart",
    categorical_strategy="fisher",
)
```

Categorical models now preserve that transform contract in:

- semantic serialization
- IR export
- compiled optimized artifacts

So restored categorical models still accept raw categorical inputs rather than
requiring callers to reproduce the transform manually.

## Sample weights

ForestFire accepts per-row training weights via `sample_weight`.

- `None` (default): all rows contribute equally
- 1-D numeric array of length `n_rows`: rows are weighted by those values

For regression, split scoring uses weighted MSE: `Σ w_i (y_i - ȳ)²`.

For gradient boosting, weights scale the per-row gradient and Hessian so high-weight rows drive larger update steps.

Weights are propagated through bootstrap sampling in `rf` and through gradient-focus row sampling in `gbm`, so ensemble methods respect weights correctly.

## Multi-target regression

When `y` is a 2-D array of shape `(n_rows, n_targets)` with `n_targets > 1`
and `task="regression"`, a single tree is trained to predict all targets
jointly.

- splits are chosen by maximising the sum of MSE gain across all targets, evaluated at a shared threshold per feature so score and applied split are always consistent
- each leaf stores one predicted value per target
- `predict(...)` returns a 2-D array of shape `(n_rows, n_targets)`
- compatible with `sample_weight`

Multi-target training uses `algorithm="dt"`. Multi-target support for `rf` and
`gbm` is not currently available.

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

- standard trees stop at the current node if no acceptable real split survives canary competition
- oblivious trees stop further depth growth if no acceptable real level-split survives canary competition
- gradient boosting stops adding stages if no acceptable real root split survives canary competition
- random forests ignore canaries during tree training

By default, that competition is strict:

- `filter=None` behaves like `filter=1`
- the top-ranked candidate must already be a real feature

You can soften that rule with `filter=...`:

- `filter=3` means the chosen real split must appear within the top 3 scored candidates
- `filter=0.95` means the chosen real split must appear within the top 5% of scored candidates

In both cases, candidates are still scored and sorted in the usual way first. The difference is only which ranked window is allowed to contain the chosen real feature.

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
- if no real root split survives inside the allowed canary `filter` window, that stage is discarded and boosting stops

That choice reflects the project’s general design preference: use explicit training-time noise competition as a stopping signal instead of bolting on a separate pruning or early-stopping layer later.

### How second-order tree fitting works

The boosting stage loop is serial, but the tree builder inside one stage is now
organized so it can become more parallel internally.

That distinction matters:

- stage `k + 1` depends on the predictions produced after stage `k`
- the nodes inside the single tree fitted at stage `k` do not all depend on one
  another in the same way

ForestFire therefore treats "fit one boosting stage" and "grow one tree inside
that stage" as separate scheduling problems.

At a high level, one boosting stage does this:

1. compute per-row gradients and Hessians from the current ensemble prediction
2. fit one second-order tree against those gradient/Hessian pairs
3. scale that tree by `learning_rate`
4. add it to the ensemble
5. repeat until the stopping rule rejects the next stage

The second-order tree itself uses Newton-style leaf values. For any node, the
leaf prediction is derived from the node-local gradient and Hessian totals:

- `leaf_value = -G / (H + lambda)`

where:

- `G` is the sum of gradients in the node
- `H` is the sum of Hessians in the node
- `lambda` is the L2 regularization term

The split score asks whether dividing the current rows into left and right
children increases the regularized objective strength enough to justify growing
the tree.

### The active-node frontier

CART and randomized trees — regression, classification, and gradient boosting
alike — now grow with an active-node frontier instead of depth-first recursion.

The frontier is simply the set of nodes at the current depth that are still
eligible to split.

That means tree growth now looks like this:

1. start with the root in the frontier
2. evaluate every node in that frontier
3. decide which nodes actually split
4. partition rows for those winning splits
5. create all children
6. use those children as the next frontier

This is different from the older depth-first pattern:

1. evaluate one node
2. partition its rows immediately
3. recurse into the left child
4. recurse into the right child
5. only later return to same-depth siblings

The frontier-based layout is important because it separates node evaluation
from row-buffer mutation.

In the standard binary tree path, ForestFire still uses one shared mutable
row-index buffer rather than copying rows into separate node-local allocations.
Each active node owns a contiguous slice of that buffer. During evaluation, the
builder reads that slice, computes node statistics, constructs or reuses
histograms, and scores candidate features. It does not mutate the slice yet.

Only after the whole current frontier has been evaluated does the builder begin
partitioning rows for the nodes that actually split.

That separation is the architectural point of the frontier:

- evaluation is "read the current node state and score split candidates"
- partitioning is "rewrite the shared row-index slice so left/right children
  occupy contiguous ranges"
- child creation is "record those new ranges as the next active frontier"

The standard CART/randomized path now uses that boundary in three explicit
phases for each depth:

1. evaluate the current frontier in parallel
2. partition row ranges for the nodes that actually split in parallel across
   disjoint row-buffer slices
3. build child histograms for the next frontier in parallel

That means the frontier is no longer only a structural preparation for future
parallelism. It is now the execution model used by all standard CART/randomized
builders: regression, classification, and gradient boosting.

### What is evaluated for each active node

For every active node, the builder computes:

- sample count
- gradient sum
- Hessian sum
- the node leaf prediction
- the node objective strength before splitting
- feature histograms for the node rows
- the best surviving split candidate after canary filtering

The stopping checks happen at this evaluation stage:

- max depth
- `min_samples_split`
- non-positive or too-small Hessian mass
- no valid feature split after `min_samples_leaf` and
  `min_sum_hessian_in_leaf`
- canary competition blocking the node at the root

If a node does not produce a valid split, it becomes a leaf and never enters
the next frontier.

If it does split, the builder retains enough information to mutate the row
buffer afterward without rescoring the node.

When training parallelism is enabled, that evaluation phase runs across the
whole active frontier at once. Each node still scores its own candidate
features using the same histogram-based split logic as before, but same-depth
nodes can now do that work concurrently.

The later mutation phase is now parallel too. Pending same-depth splits are
sorted by their owned row ranges, then partitioned through a recursive
divide-and-conquer pass over disjoint slices of the shared row-index buffer.
That keeps the in-place row-buffer model intact while allowing non-overlapping
partitions to run concurrently.

Oblique splitting now participates in that same frontier flow for standard
`cart` and `randomized` trees across all learner families.

In that mode:

- axis-aligned and oblique candidates are ranked together at each active node
- oblique candidates are still sparse pairwise linear splits
- all candidate feature pairs available at the node are considered
- per-feature missing routing is learned for the two participating features and
  replayed at inference time

### Histograms and child reuse

The second-order tree path is histogram-based.

For each feature and node, the histogram stores:

- row count
- gradient sum
- Hessian sum

Those histograms are what let the builder score candidate thresholds without
rescanning raw rows for every possible split.

When a node is partitioned, the implementation tries to avoid rebuilding both
children from scratch:

- it builds histograms for the smaller child directly from that child’s rows
- it derives the sibling histograms by subtracting the smaller child from the
  parent histograms

That matters because histogram construction is one of the dominant training
costs. Reusing the parent histogram avoids paying that full cost twice per
split.

That reuse now happens inside the frontier pipeline as well. After partitioning
has produced the child row ranges for every winning split at the current depth,
the builder constructs the next frontier’s histograms in parallel across those
splitting nodes.

### Why the frontier matters for parallelism

The frontier provides both the execution boundary and the actual parallel work
for all standard CART/randomized trees.

Without a frontier, same-depth siblings are entangled with mutation order:

- node `A` is evaluated
- node `A` partitions the shared row buffer
- node `A` recurses into its children
- only later does node `B` get evaluated

That ordering makes same-depth batching awkward because one branch has already
started rewriting shared training state before its siblings have even been
scored.

With a frontier, every node at a depth is evaluated against the same stable
view of the current row ownership. That makes several future steps much cleaner:

- scoring active nodes in parallel
- feature-parallel split search inside each node evaluation
- partitioning disjoint same-depth row ranges in parallel once split choices
  are known
- building child histograms across splitting nodes in parallel after
  partitioning
- postponing row partitioning until after split selection is complete

For gradient boosting, the stage loop still remains serial. The frontier only
changes the work inside one stage’s tree fit.

That is the intended balance:

- keep stage semantics simple and correct
- make intra-tree work increasingly parallel over time

### Relationship to oblivious trees

Oblivious second-order trees were already level-wise by construction because
every node at a given depth shares the same feature/threshold pair.

The active-node frontier brings all standard CART/randomized paths closer to
that same batching style, but without changing the standard-tree semantics:

- standard trees still choose splits independently per node
- oblivious trees still choose one shared split per depth

What they share is the key scheduling idea that depth-level work can be
evaluated as a batch before the next level is created.

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
- all standard CART/randomized trees (regression, classification, and
  gradient boosting) now grow through a level-wise active-node frontier, with
  parallel frontier evaluation, feature-parallel split scoring, parallel row
  partitioning over disjoint slices, and parallel child-histogram construction
  separated by explicit frontier phases
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
