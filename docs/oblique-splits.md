# Oblique Splits

ForestFire supports two split families: axis-aligned and oblique.

An axis-aligned split tests one feature against one threshold:

```text
x_i <= t
```

An oblique split tests a linear combination of two features:

```text
w1 * x_i + w2 * x_j <= t
```

That is the entire structural difference. The rest of the training pipeline, canaries, missing-value routing, candidate ranking, and ensemble integration, stays the same.

## Why oblique splits exist

Axis-aligned splits can only draw boundaries parallel to the feature axes.

That is a strong constraint. Many real-world patterns produce decision surfaces that are rotated or diagonal in feature space.

Consider a dataset where the true rule is:

```text
x_0 + x_1 > 0
```

An axis-aligned tree cannot express that boundary in a single node. It must approximate it with a staircase of many axis-aligned cuts: first one threshold on `x_0`, then one on `x_1`, then deeper cuts to refine the step pattern.

A single oblique node can capture the same boundary directly:

```text
1.0 * x_0 + 1.0 * x_1 <= 0
```

In those settings, oblique splits produce shallower, more expressive trees. The same tree depth covers more of the true function structure.

## How weights are computed

The weights `w1` and `w2` are learned from the training rows that reach the node. They are not user-supplied or randomly initialized.

ForestFire uses different approaches depending on the task.

### Classification

For a given feature pair `(x_i, x_j)`, ForestFire looks for the direction in that 2D feature subspace that best separates the observed classes.

The procedure is:

1. Compute the centroid of each class in the `(x_i, x_j)` plane using the rows at the current node.
2. Find the class pair whose centroids are furthest apart in Euclidean distance.
3. Use the vector from one centroid to the other as the raw weight direction `(dx, dy)`.
4. Normalize to unit length: `w1 = dx / norm`, `w2 = dy / norm`.

This is a discriminant-direction heuristic. It projects rows along the axis that maximally separates the most distant class pair, then scores all possible thresholds along that axis.

### Regression

For regression, ForestFire uses the covariance of each feature with the target.

The procedure is:

1. For each feature in the pair, compute `cov(x_k, y)` over the rows at the current node using mean-centered values.
2. Treat those two covariances as the raw weight vector `(w1_raw, w2_raw)`.
3. Normalize to unit length.

This finds the direction in the 2D subspace that is most correlated with the target, which is the direction of steepest gradient in that feature plane.

### Second-order (GBM)

For second-order gradient boosting trees, the same gradient-direction approach applies but uses per-row gradients and Hessians from the current ensemble state rather than raw target values.

### Threshold selection

Once weights are fixed for a candidate pair, ForestFire:

1. Projects each node row with complete values onto the learned direction: `value = w1 * x_i + w2 * x_j`.
2. Sorts those projected values.
3. Evaluates midpoints between consecutive distinct projected values as candidate thresholds.
4. Chooses the threshold that produces the best split score.

This is the same threshold-scanning logic used for axis-aligned splits, applied to the projected 1D space.

## Candidate enumeration

For a node with `k` candidate features, ForestFire generates all unordered pairs. That is `k * (k - 1) / 2` candidate pairs.

Each pair generates one oblique candidate using its independently computed weight vector and best threshold. Those oblique candidates join the pool of axis-aligned candidates.

Example: if `max_features=4` at a node, there are `6` axis-aligned candidates and `6` oblique candidates for a total pool of `12` before canary competition.

## Competition with axis-aligned candidates

Oblique and axis-aligned candidates compete in the same ranked pool.

Both are scored by the same objective: Gini impurity reduction for classification, MSE reduction for regression, or the second-order gain criterion for boosting. The candidate with the best score wins the node, regardless of split type.

No preference is given to either family. An oblique split wins only if its projected linear boundary produces a better score than any single-feature threshold at the same node.

That means oblique splits do not add a "bias toward complex splits." They compete on the same footing. When the true signal is axis-aligned, axis-aligned candidates typically win. When the signal is better captured by a linear combination, oblique candidates do.

## Canary interaction

Oblique splits interact with canaries the same way axis-aligned splits do.

For each real feature pair `(x_i, x_j)`, ForestFire generates a corresponding canary pair using shuffled copies of those two features. That canary pair enters the competition pool alongside the real oblique and real axis-aligned candidates.

The canary acceptance rule is identical:

- candidates are scored and sorted
- the `filter` window determines how far down the ranked list the chosen real split can fall
- if no real candidate (axis-aligned or oblique) survives inside the allowed window, the node stops

This keeps the same "better than structured noise" guarantee for both split families. An oblique node that would only win against canaries is treated as no signal, not as a special case.

For the full canary algorithm, see [Canary Strategy](canaries.md).

## Missing-value handling

Oblique splits handle missing values per participating feature rather than as a single node-level fallback.

When a row arrives at an oblique node and one of its two participating features is missing, ForestFire routes it using the missing-direction learned for that specific feature. When both features are missing and the two learned directions disagree, the tie is broken by the feature with the larger absolute weight.

For the full missing-routing semantics and decision rules, see [Missing-Value Handling: Oblique split behavior](missing-values.md#oblique-split-behavior).

## Support matrix

- `split_strategy="axis_aligned"`: supported for all tree families and algorithms
- `split_strategy="oblique"`: supported for `dt`, `rf`, and `gbm` when `tree_type` is `cart` or `randomized`

Not currently supported for:

- `id3` and `c45`: these are multiway classifiers. Their split model is not a binary threshold over one scalar, so pairwise linear splits do not compose with their branch structure.
- `oblivious`: oblivious trees share one split per depth across all nodes at that level. A single shared oblique split with learned pair-specific weights does not map cleanly onto that level-sharing model.

## Current limitations

- **Pairwise only.** The current implementation is limited to exactly two features per oblique node. Arbitrary `k`-feature hyperplanes are not yet supported.
- **All pairs considered.** Every candidate pair at a node is always evaluated. There is no adaptive search budget or pair-pruning heuristic for large feature spaces yet.
- **Row-wise inference.** The optimized runtime for oblique nodes uses a standard row-wise dot-product evaluation. A batched or SIMD-accelerated projection path is planned but not yet implemented.

## When to use oblique splits

Use oblique splits when the problem has known or suspected structure across feature pairs:

- numeric features that are correlated and jointly predictive
- decision surfaces that are rotated relative to the feature axes
- tasks where shallower, more expressive trees are preferred over many small axis-aligned fragments

Oblique splits are also useful inside gradient boosting. Each stage fits trees to residual structure, and residual structure can be diagonal in feature space even when the original target was not.

Start with axis-aligned splits for a baseline. Try oblique splits when:

- axis-aligned trees are growing very deep on structured data
- you believe pairs of features interact in ways that a single threshold misses
- you want to compare shallower trees against the axis-aligned baseline

### Tradeoffs

**Training cost:** Every candidate feature pair generates one oblique candidate. For a node with `k` candidate features, this multiplies the candidate pool. On large feature spaces, training with oblique splits can be substantially slower than axis-aligned training.

**Inference cost:** Each oblique node requires a dot product plus one threshold comparison rather than one threshold comparison alone. This is a modest constant-factor overhead per node, not a change in asymptotic complexity.

**Interpretability:** Each oblique node depends on two features. A node condition like `0.71 * x_3 + 0.71 * x_7 <= 0.4` is harder to read than `x_3 <= 0.4`. For use cases where human-readable trees matter, this is a real tradeoff.

**When axis-aligned wins anyway:** If the true decision boundary is axis-aligned, oblique candidates still compete but do not win. The competition is fair; the cost is the extra computation for generating and scoring those candidates before they lose.

## Usage examples

### Decision tree with oblique splits

```python
import numpy as np
from forestfire import train

rng = np.random.default_rng(42)
n = 10_000

# Rotated boundary: the rule is x_0 + x_1 > 0.5
X = rng.normal(size=(n, 6))
y = (X[:, 0] + X[:, 1] > 0.5).astype(float)

# Axis-aligned baseline: needs many nodes to approximate the diagonal boundary
model_aa = train(
    X,
    y,
    algorithm="dt",
    tree_type="cart",
    task="classification",
)

# Oblique: the diagonal boundary can be expressed in far fewer nodes
model_ob = train(
    X,
    y,
    algorithm="dt",
    tree_type="cart",
    task="classification",
    split_strategy="oblique",
)

print(model_aa.tree_structure())
print(model_ob.tree_structure())
```

### Random forest with oblique splits

```python
model_rf = train(
    X,
    y,
    algorithm="rf",
    tree_type="cart",
    task="classification",
    split_strategy="oblique",
    n_trees=200,
    max_features="sqrt",
)
```

### Gradient boosting with oblique splits

```python
model_gbm = train(
    X,
    y,
    algorithm="gbm",
    tree_type="cart",
    task="classification",
    split_strategy="oblique",
    learning_rate=0.05,
    n_trees=500,
    canaries=2,
    filter=0.95,
)
```

### Inspecting oblique nodes

Oblique split nodes appear in tree introspection alongside axis-aligned nodes. The `tree_structure()` and `tree_node()` methods expose them using the same interface, but the split description now includes two feature indices and their corresponding weights.

```python
model = train(
    X,
    y,
    algorithm="dt",
    tree_type="cart",
    task="classification",
    split_strategy="oblique",
)

print(model.tree_structure())
print(model.tree_node(node_id=0, tree_index=0))

ir = model.to_ir_json()
```

In the IR, oblique nodes appear as `ObliqueLinearCombination` splits with:

- `feature_indices`: the two participating feature positions
- `weights`: the two normalized weight values
- `threshold`: the learned threshold in projected space
- `missing_directions`: the per-feature learned missing routing

### Oblique splits with randomized trees

Randomized tree families introduce stochasticity into split search. Oblique splits compose naturally with that: the randomized tree still evaluates multiple threshold candidates per pair, with oblique and axis-aligned candidates competing in the usual way.

```python
model_rand_ob = train(
    X,
    y,
    algorithm="dt",
    tree_type="randomized",
    task="classification",
    split_strategy="oblique",
    seed=7,
)
```

## Why the implementation is pairwise

Extending to arbitrary `k`-feature hyperplanes would allow the split to express any linear boundary, not just one between two features. That is a more powerful hypothesis class.

The reasons for starting at pairs are:

- **Search cost.** For `k` selected features, all unordered pairs is `O(k^2)`. All `m`-feature subsets for `m > 2` grows combinatorially. Without a principled search heuristic, the candidate space becomes impractical.
- **Weight estimation.** The current weight estimation procedure is closed-form and cheap. Estimating the best direction for `m > 2` features well requires either more data per node or a more expensive optimization step.
- **Interpretability floor.** Two-feature oblique nodes are already less interpretable than single-feature thresholds. Going beyond pairs reduces interpretability further without a compelling accuracy justification for most tabular problems.

Pairwise oblique splits hit a useful point in the tradeoff space:

- they capture the most common multi-feature structure (correlated pairs and diagonal boundaries)
- without an exponential search cost or complex weight estimation
- while remaining reasonably close to what a practitioner can inspect

The roadmap includes moving beyond pairwise splits as optional experimental extension once adaptive search budgeting and stronger pair-selection heuristics are in place.
