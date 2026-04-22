# Missing-Value Handling

ForestFire treats missing values as a first-class part of tree semantics rather
than as a preprocessing afterthought.

This applies across:

- single trees
- random forests
- gradient boosting
- exact semantic prediction
- optimized inference

The input-side missing markers can come from several Python ecosystems:

- Python `None`
- floating-point `NaN`
- pandas / NumPy `NaN`
- `polars` null values

## Core idea

Every feature gets an explicit missing bucket in its binned representation.

That means the trainer does not need a separate “impute first” phase just to
make split search possible. Missingness is part of the search space.

The high-level rule is:

1. build histograms with one dedicated missing bin
2. ignore that missing bin when choosing the observed split structure
3. decide where the missing rows should go once a split candidate is being evaluated

That gives ForestFire a clean separation between two different questions:

- what is the best split over observed values?
- if a row is missing at this node, should it go left, right, or fall back to the node prediction?

## Training-time split semantics

For numeric features:

- observed values live in the ordinary numeric bins
- missing values live in one extra missing bin
- threshold search only considers the observed bins

For binary features:

- `false`
- `true`
- `missing`

Again, the missing bucket is tracked explicitly instead of being coerced into
either observed branch up front.

## Missing-value strategies

ForestFire currently exposes two split-search strategies:

- `heuristic`
- `optimal`

### `heuristic`

Under the heuristic strategy:

1. choose the best split using observed values only
2. once that split is known, score the result with missing rows sent left
3. score it again with missing rows sent right
4. keep the better of those two routings

Why this exists:

- it keeps split search fast
- it avoids expanding the full candidate space
- it preserves explicit missing routing at the learned node

This is the default strategy.

### `optimal`

Under the optimal strategy:

- every candidate split is evaluated together with missing-left and
  missing-right routing
- the chosen result is the best joint combination of observed split plus
  missing routing

Why this exists:

- it is semantically cleaner
- it can find better splits when missingness itself carries strong signal

Tradeoff:

- it is substantially slower because the search space is larger

### Per-feature strategy selection

The Python API also allows the strategy to vary by column.

That matters because missingness is often uneven across a table:

- some features are densely populated and do not need expensive missing search
- some features are sparse enough that full missing-path optimization is worth it

So the strategy configuration is part of the tree-building semantics, not just
an implementation detail.

## What a trained node stores

When a split actually observes missing values during training, the learned node
stores a missing-direction decision:

- missing goes left
- missing goes right

That decision becomes part of prediction semantics for that node.

If the feature had no missing values at training time for that split, the node
does not pretend that a missing branch was learned. In that case, a later
missing inference value falls back to the node prediction.

That fallback is:

- majority class / node probabilities for classification
- node mean prediction for regression

This is deliberate. It avoids inventing a synthetic missing branch that the
trainer never had evidence to prefer.

## Why this design was chosen

ForestFire does not use “always send missing left” or “always send missing
right” as a global convention.

That kind of fixed convention is cheap, but it hard-codes an execution rule
that may have nothing to do with the actual node statistics.

ForestFire also does not force global imputation before training, because that
would erase potentially useful signal:

- “this value is missing” can itself be predictive
- different nodes can legitimately prefer different missing directions

The chosen design keeps missingness inside the tree learner:

- preprocessing records it
- histograms represent it
- split scoring reasons about it
- prediction reproduces the learned routing

## Oblique split behavior

Oblique splits now follow the same general missing-value principle, but they
need one extra rule because two features participate in the same node.

An oblique split currently has the form:

```text
w1 * x_i + w2 * x_j <= t
```

For those nodes, missing values are handled per participating feature rather
than as one undifferentiated “oblique node is missing” case.

That means:

- feature `x_i` learns its own missing direction
- feature `x_j` learns its own missing direction
- a row missing only one of those features is routed using that feature’s
  learned direction

If both participating features are missing:

- the node first checks whether the two learned directions agree
- if they do, it follows that shared direction
- if they disagree, the tie is resolved by the feature with the larger absolute
  oblique weight

So oblique missing routing is still explicit learned tree semantics, not a
serving-time imputation trick.

## Relation to optimized inference

Optimized runtimes preserve the same missing-value semantics as the semantic
model.

There is one extra optimization knob:

- users can specify which features should retain missing checks in the optimized
  runtime

That is useful when:

- a model was trained with missing-aware semantics
- but the deployment pipeline guarantees that some columns will never be
  missing at inference time

In that case, missing checks for those features can be removed from the lowered
runtime without changing the expected deployment semantics.

## Current implementation boundary

The first-order tree paths implement the configurable missing-value strategy
surface directly.

The second-order boosting path still uses the existing missing-value behavior
internally rather than a fully separate heuristic-vs-optimal strategy choice.

That means the public missing-value semantics are aligned across the library,
but the configurable search strategy is not yet equally rich in every internal
trainer.

For oblique splits specifically, the current implementation is:

- first-order trees: learned per-feature missing directions
- second-order GBM trees: learned per-feature missing directions as well
- configurable `heuristic` vs `optimal` missing strategy: still a first-order
  tree-builder setting rather than a separate GBM toggle
