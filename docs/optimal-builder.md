# Optimal Builder

The optimal builder is an exhaustive tree-construction strategy.

Unlike `greedy`, `lookahead`, and `beam`, it does not use explicit builder-side
limits such as a fixed rescoring horizon, shortlist cap, beam width, or future
weight. Instead, it evaluates the full downstream subtree objective for every
legal split that survives ordinary split generation.

## Public API

Python:

```python
train(
    X,
    y,
    builder="optimal",
)
```

Rust:

```rust
use forestfire_core::{BuilderStrategy, TrainConfig};

let config = TrainConfig {
    builder: BuilderStrategy::Optimal,
    ..TrainConfig::default()
};
```

## How it works

At each node:

1. score all legal split candidates
2. recursively evaluate every candidate's full descendant objective
3. stop descending a branch when:
   - `max_depth` is reached
   - the node is too small to split
   - the target is already pure / constant
   - second-order statistics become invalid
   - canary filtering blocks all acceptable real splits at that node
4. choose the best surviving real split under that full recursive score

This makes `optimal` a true subtree search rather than a bounded rescoring pass.

## Parameters

- `builder="optimal"` enables exhaustive subtree search.
- `max_depth`, `min_samples_split`, `min_samples_leaf`, and canary filtering are
  the meaningful controls over search size and stopping.

The following builder-tuning knobs are ignored by `optimal`:

- `lookahead_depth`
- `lookahead_top_k`
- `lookahead_weight`
- `beam_width`

## Missing values

Like the other builders, `optimal` uses the learner’s actual missing-value
semantics while evaluating branches:

- learned missing routing for axis-aligned trees
- per-feature missing directions for oblique trees
- multiway missing handling for `id3` and `c45`
- current second-order GBM missing routing

## Support matrix

`builder="optimal"` is exposed anywhere the corresponding learner family is
already supported:

- decision trees
- random forests
- gradient boosting

and across the current tree-family surface:

- `id3`
- `c45`
- `cart`
- `randomized`
- `oblivious`

## Tradeoffs

Why use it:

- it is the least myopic builder currently available
- it lets canary stopping and structural constraints define the actual search
  boundary instead of a fixed lookahead horizon

Costs:

- it is usually the slowest builder
- search cost can grow very quickly with depth and candidate count
- practical use depends heavily on canaries and depth limits to prune the tree

Reasonable first settings:

- keep `max_depth` small
- keep canaries/filtering enabled
- start on smaller feature sets before scaling up
