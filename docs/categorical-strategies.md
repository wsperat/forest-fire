# Categorical Strategies

ForestFire currently supports three practical categorical strategies through
the normal native `train(...)` interface:

- `dummy`
- `target`
- `fisher`

These strategies are now implemented in Rust, not as Python-side
preprocessing. Python sends mixed rows directly into the native training path,
and Rust applies the configured categorical transform before handing the
resulting representation to the tree/forest/GBM trainers.

The implementation is intentionally split by responsibility:

- `dummy` and `target` live close to the table layer because they are
  fundamentally table-to-table feature transforms
- `fisher` lives in core because it is not just a generic encoding; it is a
  target-informed category-ordering strategy tied to tree split semantics

That said, the current system is still based on transformed numeric/binary
features rather than fully native categorical predicates in the IR and runtime.

This page explains what each strategy means, when it is appropriate, and what
its tradeoffs are.

## Why categorical handling is a separate topic

Tree libraries often make a sharp distinction between:

- numeric features
- binary features
- categorical features

That distinction matters because categories are labels, not magnitudes.

If a column contains values such as:

- `red`
- `blue`
- `green`

then the learner should not assume:

- `green > blue`
- `blue` is “between” `red` and `green`
- adjacent integer ids imply semantic similarity

The central problem is therefore not just storage. It is split semantics.

Different strategies solve that problem in different ways:

- `dummy` expands one categorical source column into several binary indicators
- `target` replaces each category with a target-derived statistic
- `fisher` orders categories by target-derived statistics and then treats that
  order as a numeric feature for threshold splits

None of those is universally correct. They are practical approximations with
different bias and variance behavior.

## Strategy Selection

The rough rule of thumb is:

- use `dummy` for low-cardinality nominal features
- use `target` for higher-cardinality features when width growth would become
  expensive
- use `fisher` when you want a LightGBM-like ordered-category split surrogate
  for binary-threshold tree learners

If you are unsure:

- start with `dummy` for small vocabularies
- consider `target` or `fisher` once category counts become large enough that
  width expansion is no longer attractive

## `dummy`

`dummy` is the renamed public term for the classic one-of-K compatibility
encoding.

For a source column like:

- `red`
- `blue`
- `green`

the encoded representation becomes a set of binary features:

- `is_red`
- `is_blue`
- `is_green`
- plus an extra unknown/missing indicator in the current implementation

So a row with category `blue` becomes something like:

- `0, 1, 0, 0`

and an unseen or missing category becomes:

- `0, 0, 0, 1`

### Why the name `dummy`

The term “dummy variable” is standard statistical language, and it avoids
mixing the public API name with the internal implementation detail that the
expansion is effectively one-hot-like.

### Why `dummy` is useful

This strategy is the lowest-risk way to add categorical support to existing
binary split learners because it reuses machinery that already works well:

- binary feature handling
- binary histogram construction
- ordinary threshold or boolean split logic

That makes it a strong baseline when:

- categories are nominal
- cardinality is low
- interpretability of individual category indicators is useful

### Strengths

- semantically safe for nominal categories
- easy to reason about
- low implementation risk
- naturally compatible with existing binary-feature tree code
- explicit unknown-category fallback can be represented as its own indicator

### Weaknesses

- feature width grows with category count
- `max_features` behavior can get distorted because one source feature becomes
  many derived columns
- grouped category rules may require multiple tree levels

That last point matters. A native categorical split could represent:

- `red or green -> left`
- `blue -> right`

in one node.

A dummy encoding often needs multiple binary tests to recover the same logic.

### Best use cases

- small vocabularies
- mostly nominal categories
- compatibility-oriented first pass
- cases where model width growth is acceptable

## `target`

`target` encoding replaces each category with a statistic derived from the
training target.

Conceptually:

- for regression, categories are mapped to smoothed target means
- for classification, categories are mapped to smoothed class probabilities

So instead of turning a feature with 500 categories into 500 indicator columns,
the strategy turns it into one or a few numeric columns.

### Why this helps

This is attractive when cardinality is high because the representation size no
longer grows linearly with the number of categories.

That means:

- narrower feature matrices
- less pressure on feature subsampling
- simpler downstream split search

### Smoothing

Naive target encoding is dangerous because rare categories can overfit badly.

If a category appears once and that row’s target is positive, a naive encoding
would treat that category as maximally positive. That is usually too confident.

ForestFire’s current `target` strategy therefore uses smoothing toward a global
prior.

Conceptually the encoded value is:

- a blend of the category-local estimate
- and the overall target prior

The `target_smoothing` parameter controls how strongly rare categories are
pulled back toward the global prior.

Larger smoothing means:

- more shrinkage toward the global average
- less variance on rare categories
- more bias for legitimately strong but infrequent levels

Smaller smoothing means:

- more category-specific flexibility
- more sensitivity to rare-level noise

This parameter is part of categorical modeling itself, not part of the canary
policy.

In other words:

- `target_smoothing` defines the base amount of shrinkage used by categorical
  training
- canaries may later increase the effective smoothing for `target` and
  `fisher` when shuffled-category signal is too competitive

So if you are trying to understand what the parameter means, this is the
primary section to read. The canaries page only explains when ForestFire may
adapt that base value upward for robustness.

### Classification behavior

For classification, the encoder currently emits one value per class. That means
a categorical feature can expand to:

- one encoded column in binary classification if the downstream path collapses
  it further later
- or several numeric columns representing class-probability-like statistics

In the current implementation, these encoded values are then treated as numeric
features by the tree learner.

### Strengths

- scales much better than `dummy` on high-cardinality features
- can capture useful target associations compactly
- unknown categories can fall back cleanly to the global prior

### Weaknesses

- target leakage is a real conceptual risk in general target encoding
- encoded values depend on the training target, not just feature structure
- category meaning becomes less directly interpretable

The current implementation is a practical smoothed training-time encoding, not
a full cross-fitting or CatBoost-style ordered-statistics system.

That means it is useful now, but it should be understood as an initial
pragmatic implementation rather than the final leakage-minimizing design.

### Best use cases

- medium or high-cardinality categorical columns
- regression problems
- classification problems where width growth would become too large under
  `dummy`
- cases where compactness matters more than strict interpretability of raw
  category rules

## `fisher`

`fisher` is an ordered-category strategy inspired by the category ordering idea
used by systems like LightGBM for histogram-based categorical split search.

The basic idea is:

1. compute target-derived statistics per category
2. order categories by those statistics
3. assign each category a numeric rank
4. let the tree learner split over that ordered axis with ordinary threshold
   logic

So categories are not expanded into many columns. They become a single ordered
numeric surrogate feature.

### Why this is useful

This gives binary split learners a way to separate categories using one
threshold over a learned ordering rather than many indicator columns.

That can be much more compact than `dummy` while still behaving more like a
categorical split than arbitrary integer ids would.

The important distinction is that the order is not user-supplied or arbitrary.
It is derived from target statistics.

That is what makes this strategy categorically motivated rather than just naive
ordinal encoding.

### Why “Fisher”

The name here is meant to describe category separation by target-informed
ordering and threshold scan, not to claim a full clone of any single external
library’s exact categorical algorithm.

The current implementation uses target-derived category ordering and then maps
that order to ranks consumed by the existing numeric split machinery.

Although `fisher` is not itself target encoding, it still depends on
target-derived category statistics. For that reason, the same
`target_smoothing` parameter is also used as the base regularization level for
the per-category statistics that determine the ordering.

### Strengths

- compact representation
- often more expressive than `dummy` for medium-cardinality categoricals
- closer in spirit to native categorical threshold ordering than arbitrary
  integer labels
- fits naturally into binary threshold tree learners

### Weaknesses

- still not a fully native categorical subset split
- quality depends on how informative the target-derived ordering is
- can be unstable when categories are extremely rare
- like `target`, it depends on target-derived statistics

### Best use cases

- medium-cardinality categorical features
- boosted trees and other binary split learners where grouped category
  separation matters
- cases where `dummy` would be too wide but full native categorical support is
  not available yet

## Unknown And Missing Categories

Current behavior depends on the strategy:

- `dummy`: unseen or missing categories activate the dedicated unknown/missing
  indicator
- `target`: unseen or missing categories fall back to the global prior
- `fisher`: unseen or missing categories fall back to `NaN`, which then follows
  the library’s existing missing-value handling

That means unknown-category behavior is explicit rather than silently collapsing
into one observed category.

## Interaction With ForestFire Training

After categorical handling runs, the downstream trainers still see ordinary
numeric or binary feature matrices.

So all existing training behavior still applies:

- trees, forests, and GBM all use the same transformed inputs
- canaries still compete against transformed features
- histogram-based split search still works the same way after encoding
- missing-value handling still follows the numeric/binary semantics of the
  transformed representation

This is both the strength and the limitation of the current design:

- it makes categorical strategies available through the same native training
  path as the rest of the library
- but the model is still learning on transformed features, not on native
  categorical predicates

At the API level, this means there is no separate categorical training entry
point. You still call `train(...)` and pass categorical options directly:

```python
train(
    X,
    y,
    task="classification",
    tree_type="cart",
    categorical_strategy="dummy",
)
```

The same applies to the sklearn-style wrappers, which forward the categorical
configuration through the same native path.

## Current Limits

The current categorical strategies are native Rust transforms, but they are not
yet represented as first-class categorical semantics in the serialized/IR model
contract.

That means:

- IR export is disabled for models trained with categorical transforms
- serialization is disabled for those models
- compiled optimized serialization is disabled for those models

The reason is straightforward: the Rust semantic model and IR do not yet record
the categorical transform contract.

Until they do, serialization would not be faithfully reproducible.

## Practical Guidance

If you want a simple starting point:

- use `dummy` when category counts are small

If width becomes a problem:

- try `target`

If you want a compact LightGBM-like category ordering surrogate for binary
split learners:

- try `fisher`

There is no universal winner. The right choice depends on:

- cardinality
- rare-category frequency
- whether interpretability matters
- whether width or leakage risk is the bigger practical concern
