# Beam Builder

The beam builder extends the lookahead idea by keeping several strong future
continuations alive instead of following only one.

This makes it a middle ground between:

- `builder="greedy"`: immediate score only
- `builder="lookahead"`: immediate score plus one best continuation
- `builder="beam"`: immediate score plus a width-limited continuation search

## Public API

Python:

```python
train(
    X,
    y,
    builder="beam",
    lookahead_depth=2,
    lookahead_top_k=8,
    lookahead_weight=0.5,
    beam_width=4,
)
```

Rust:

```rust
use forestfire_core::{BuilderStrategy, TrainConfig};

let config = TrainConfig {
    builder: BuilderStrategy::Beam,
    lookahead_depth: 2,
    lookahead_top_k: 8,
    lookahead_weight: 0.5,
    beam_width: 4,
    ..TrainConfig::default()
};
```

## Parameters

- `builder="beam"` enables width-limited continuation search.
- `lookahead_depth` controls how many future levels are considered.
- `lookahead_top_k` limits which immediate candidates are eligible for
  continuation rescoring.
- `lookahead_weight` controls how strongly future score influences ranking.
- `beam_width` controls how many descendant continuations stay alive at each
  future rescoring step.

The current ranking model is still:

```text
ranking_score = immediate_gain + lookahead_weight * future_gain
```

What changes relative to plain lookahead is how `future_gain` is estimated.

## How it works

At the current node:

1. score all candidates by immediate gain
2. keep the top `lookahead_top_k`
3. partition rows for each shortlisted candidate
4. score the next level recursively
5. at each future step, keep the top `beam_width` ranked continuations alive
6. use the strongest surviving continuation score as the candidate's
   `future_gain`
7. choose the final winner after normal canary filtering

So the current implementation is a width-limited continuation search layered on
top of local split ranking. It is not yet a full global beam search over whole
partial-tree states.

That distinction matters:

- the beam builder improves local split choice
- it does not yet optimize the entire tree jointly
- it remains compatible with the existing tree-construction code paths

## Missing values

Like the lookahead builder, the beam builder uses the learner’s existing
missing-value semantics during future rescoring.

That means the simulated continuation search respects:

- learned missing routing for axis-aligned splits
- per-feature missing directions for oblique splits
- multiway missing handling for `id3` and `c45`
- current second-order GBM missing routing

The future score is therefore based on the same partition behavior the final
tree would use if that candidate wins.

## Support matrix

`builder="beam"` is exposed anywhere the corresponding learner family is
currently supported:

- decision trees
- random forests
- gradient boosting

and across the current tree-family surface:

- `id3`
- `c45`
- `cart`
- `randomized`
- `oblivious`

Split-strategy support still applies independently. In particular, oblique
splits remain limited to the tree families that already support
`split_strategy="oblique"`.

## Tradeoffs

Why use it:

- it is less short-sighted than greedy scoring
- it is more robust than single-path lookahead when several future continuations
  look plausible
- it can recover from locally ambiguous node choices better than plain
  lookahead

Costs:

- slower than both `greedy` and `lookahead`
- higher memory and ranking overhead during rescoring
- wide beams can spend effort preserving branches that do not ultimately matter

Reasonable first settings:

- `lookahead_depth=2`
- `lookahead_top_k=4` or `8`
- `lookahead_weight=0.25` to `0.75`
- `beam_width=2` or `4`

## When to use beam over lookahead

Prefer `beam` when:

- the node has several similarly strong immediate splits
- you suspect the best split is only obvious after more than one plausible
  continuation is explored
- you can afford extra training time

Prefer `lookahead` when:

- you want a cheaper upgrade over greedy search
- training cost matters more than squeezing out a better local choice
- the data usually has one clearly dominant continuation anyway
