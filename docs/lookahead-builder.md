# Lookahead Builder

The lookahead builder is an alternative to the default greedy tree-construction
path.

With the greedy builder, a node picks the split with the best immediate score at
that node and commits to it immediately.

With the lookahead builder, a node still starts from immediate scores, but it
re-ranks a shortlisted set of candidates by estimating how much useful
downstream structure they expose in their children.

Operationally, `lookahead` is the same recursive subtree scorer used by
`optimal`, but with explicit limits applied:

- recursion stops after `lookahead_depth`
- only the top `lookahead_top_k` immediate candidates are rescored
- future score is scaled by `lookahead_weight`

## Public API

Python:

```python
train(
    X,
    y,
    builder="lookahead",
    lookahead_depth=2,
    lookahead_top_k=8,
    lookahead_weight=0.5,
)
```

Rust:

```rust
use forestfire_core::{BuilderStrategy, TrainConfig};

let config = TrainConfig {
    builder: BuilderStrategy::Lookahead,
    lookahead_depth: 2,
    lookahead_top_k: 8,
    lookahead_weight: 0.5,
    ..TrainConfig::default()
};
```

## Parameters

- `builder="lookahead"` enables the lookahead ranking path.
- `lookahead_depth` controls how many additional levels are explored while
  scoring a candidate split.
- `lookahead_top_k` limits how many immediate candidates are eligible for
  lookahead rescoring.
- `lookahead_weight` controls how strongly future score affects the final
  ranking.

The effective ranking is:

```text
ranking_score = immediate_gain + lookahead_weight * future_gain
```

If `lookahead_depth <= 1`, the behavior collapses back to greedy ranking.

## How it works

At each node:

1. score all legal split candidates by immediate gain
2. sort them by that immediate score
3. keep only the top `lookahead_top_k`
4. for each shortlisted candidate, simulate the child partitions
5. score the best child continuation recursively up to `lookahead_depth`
6. combine immediate and future score into one ranking value
7. choose the highest-ranked real split after canary filtering

This means the builder is still fundamentally local: it does not optimize the
whole tree jointly. What changes is the scoring rule for a candidate split at
the current node.

## Missing values

Lookahead does not invent a separate missing-value policy. It uses the exact
missing-value behavior of the underlying learner while simulating the future
partitions.

That means:

- axis-aligned trees reuse their learned missing-branch routing
- oblique trees reuse their per-feature missing directions
- ID3/C4.5 reuse their multiway missing routing
- second-order GBM trees reuse their current missing-routing semantics during
  lookahead scoring

This is important because the rescoring pass should evaluate the same routing
behavior that the final built tree will actually use.

## Support matrix

`builder="lookahead"` is available anywhere the corresponding learner family is
already supported:

- decision trees
- random forests
- gradient boosting

and across the currently exposed tree families:

- `id3`
- `c45`
- `cart`
- `randomized`
- `oblivious`

The actual split family limits still apply independently. For example, oblique
splits remain restricted to the tree families that already support
`split_strategy="oblique"`.

## Tradeoffs

Why use it:

- it can avoid short-sighted splits that look good locally but isolate weak
  child structure
- it is a better fit for data where useful signal appears only after one more
  split

Costs:

- training is slower than `builder="greedy"`
- candidate rescoring allocates more work per node
- deeper lookahead horizons can amplify noise if `lookahead_weight` is too high

Reasonable first settings:

- `lookahead_depth=2`
- `lookahead_top_k=4` or `8`
- `lookahead_weight=0.25` to `0.75`

## Relationship to beam search

The lookahead builder follows only the single best continuation at each future
step.

If you want to keep multiple strong continuations alive during rescoring, use
the [Beam Builder](beam-builder.md) instead.
