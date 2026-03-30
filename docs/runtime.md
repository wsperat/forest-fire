# Runtime, Serialization, And Optimization

## Why runtime design matters in tree systems

Tree libraries often optimize only the trainer and treat prediction as “walk the same structure later”.

ForestFire does not do that.

The project assumes training and inference have different performance needs:

- training tolerates richer bookkeeping
- inference cares about compact layouts, regular access patterns, and batched execution

So the runtime story is designed explicitly rather than inherited accidentally from the trainer.

## Training structure vs runtime structure

The trained model carries information that matters for understanding and exporting it:

- impurity
- gain
- sample counts
- class counts
- variances
- branch metadata

Those fields are valuable semantically but costly operationally.

On the hot prediction path they increase:

- object size
- cache pressure
- pointer chasing
- general-case branching

That is why `optimize_inference(...)` exists. It lowers the semantic model into a runtime-specialized representation.

## CART-style fallthrough layout

Binary trees are a natural fit for compact control-flow layouts.

ForestFire’s compiled CART-style runtime keeps the more common child as a fallthrough path and stores only the less common branch as an explicit jump target.

Why this helps:

- fewer explicit jumps on the hot path
- smaller runtime node representation
- less branch-heavy traversal logic

The important point is not that the model changes. It does not. The traversal strategy changes.

## Dense lookup for multiway nodes

Classifier trees with multiway branching can be expensive if every node does:

1. scan the branch list
2. compare until a matching bin is found

ForestFire lowers those into dense lookup tables indexed by bin id.

Why this helps:

- work becomes one indexed read instead of a small search
- execution becomes more regular
- the runtime cost depends less on branch-list length

This is a good example of using the regularity created by binning to simplify scoring.

## Oblivious-tree runtime design

Oblivious trees are structurally different enough that they deserve their own runtime strategy.

Because each depth shares one split, scoring can be expressed as:

- a compact array of feature indices
- a compact array of thresholds
- a loop that accumulates a leaf index

Why this matters:

- the runtime becomes array-driven rather than pointer-driven
- prediction is more regular
- SIMD- and batch-oriented execution becomes more natural

This is why oblivious trees are such a good fit for optimized runtime lowering even though they are more constrained during training.

## Feature projection

Optimized models do not eagerly preprocess every feature from the semantic input schema.

Instead, the runtime:

- scans the semantic model for the feature indices that actually appear in splits
- computes the sorted union of those indices across the whole model or ensemble
- remaps the lowered runtime into that compact local feature space
- preprocesses only the projected columns during optimized scoring

Why this helps:

- wide inputs often carry many columns the model never touches
- skipping unused columns reduces binning and packing work before traversal starts
- the benefit compounds in forests and boosted ensembles because the same projected columns are reused many times

This is a runtime optimization only. The semantic model still expects the full input schema, and both semantic and optimized models expose the same prediction behavior.

## Batched preprocessing

Prediction cost is not just model traversal. It also includes turning user input into the internal binned representation.

ForestFire therefore preprocesses batches of rows together when possible.

Why:

- per-row conversion overhead dominates small or shallow-model workloads
- batching lets the runtime reuse feature metadata and conversion logic
- compact batch formats can then be fed directly into compiled runtimes

This is also why `polars.LazyFrame` prediction is batched in chunks instead of being materialized and traversed one row at a time.

## Ensemble locality ordering

Forests and boosted ensembles also get a lightweight runtime-only ordering pass before lowering.

Today the ordering key is intentionally simple:

- primary/root split feature first
- then used-feature count
- then the full used-feature set as a stable tiebreaker

Why do this at all:

- nearby trees are more likely to touch the same projected feature columns
- top-level feature metadata and hot batch columns have a better chance of staying warm in cache
- the runtime can improve locality without changing the semantic model, IR, or predictions

This is not a semantic reorder. It only affects the lowered optimized runtime and compiled artifacts.

## Compact `u8` / `u16` batches

The runtime stores batched bin ids as:

- `u8` when the realized feature bin domain fits
- `u16` only when it has to

This is a small design detail with large practical consequences:

- less memory bandwidth
- better cache density
- more rows processed per cache line

Because training already commits to bounded bins, the runtime can aggressively exploit that compactness.

Feature projection makes this more effective:

- if the model only uses a narrow subset of columns, the compact batch matrix only stores those columns
- smaller projected matrices improve both bandwidth and cache residency
- multiway lookup tables can also be right-sized to the actual realized bin domain

## Why the IR sits in the middle

The IR is the semantic source of truth for:

- serialization
- runtime lowering
- introspection

This has two major benefits.

First, it prevents divergence:

- optimized runtime code has to lower from an explicit semantic structure
- serialization has to preserve the same explicit structure
- introspection sees the same structure instead of hidden trainer internals

Second, it makes portability realistic:

- bin boundaries are explicit
- leaf payload meaning is explicit
- model family and structure are explicit

That is what makes the project’s “train once, introspect, serialize, optimize, and score” story coherent.

## Why introspection belongs next to serialization

At first glance, introspection looks like a developer convenience while serialization looks like a deployment feature.

In practice, they need the same thing:

- a stable, explicit account of model structure

That is why ForestFire can expose:

- `tree_structure(...)`
- `tree_node(...)`
- `tree_level(...)`
- `tree_leaf(...)`
- `to_dataframe(...)`

without inventing a separate debug-only representation.

The same semantic structure used for export is what powers the model-inspection features.

## Performance impact in practical terms

The point of these runtime design decisions is not theoretical purity. It is to change where time and memory go.

Typical wins come from:

- fewer allocations on the prediction path
- more compact working sets
- more regular access patterns
- better amortization of preprocessing work across batches
- less preprocessing spent on columns the model never uses
- better feature-column reuse across ensemble members

The largest benefits generally show up when:

- the same model serves many predictions
- batch sizes are moderate or large
- trees are deep enough that traversal overhead matters
- forests or boosted ensembles amplify any per-tree inefficiency

The smallest gains generally show up when:

- batches are tiny
- models are shallow
- input conversion dominates traversal cost

That is why the project exposes optimized inference explicitly: it is a useful specialization, but not a magic speedup in every workload.
