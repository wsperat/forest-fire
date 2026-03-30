# Runtime, Serialization, And Optimization

## Why runtime design matters in tree systems

Tree libraries often optimize only the trainer and treat prediction as “walk the same structure later”.

ForestFire does not do that.

The project assumes training and inference have different performance needs:

- training tolerates richer bookkeeping
- inference cares about compact layouts, regular access patterns, and batched execution

So the runtime story is designed explicitly rather than inherited accidentally from the trainer.

The core idea is that ForestFire keeps two views of the same learned model:

- a semantic view for correctness, serialization, and introspection
- a runtime view for speed

The semantic view answers:

- what split is this node performing?
- what do the leaf values mean?
- how should outputs be postprocessed?
- what preprocessing assumptions must be preserved?

The runtime view answers:

- how should the CPU touch memory while scoring?
- which columns should be materialized first?
- which structures should be traversed row-by-row vs batch-by-batch?
- which information is hot-path critical, and which is only useful for export or inspection?

That split is what lets the project improve runtime behavior aggressively without turning serialization and introspection into moving targets.

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

That lowering step is intentionally one-way from the user’s point of view:

- training produces a semantic `Model`
- optimized inference derives an `OptimizedModel`
- serialization of the semantic model still uses the semantic IR
- compiled artifacts optionally snapshot the lowered runtime as an execution cache

This means ForestFire can answer “what does this model mean?” and “how should this model run?” separately.

## End-to-end optimized prediction lifecycle

When you call `optimize_inference(...)`, the runtime currently goes through these stages:

1. inspect the semantic model
2. compute the set of features the model actually uses
3. build a global-to-local feature remapping
4. lower each tree or ensemble member into a runtime-oriented layout
5. optionally reorder ensemble members for better locality
6. keep the semantic model around for export and inspection

When you later call `predict(...)` on the optimized model, the runtime does:

1. accept full user-facing inference input
2. validate it against the semantic preprocessing contract
3. project the input down to the used feature subset
4. bin or normalize only that projected subset
5. choose scalar or batch-oriented scoring
6. aggregate the results back into semantic predictions

That flow is the important architectural point: optimization is not just about faster tree traversal. It starts before traversal, at input materialization.

## CART-style fallthrough layout

Binary trees are a natural fit for compact control-flow layouts.

ForestFire’s compiled CART-style runtime keeps the more common child as a fallthrough path and stores only the less common branch as an explicit jump target.

Why this helps:

- fewer explicit jumps on the hot path
- smaller runtime node representation
- less branch-heavy traversal logic

The important point is not that the model changes. It does not. The traversal strategy changes.

The specific idea is:

- every binary branch has two children
- one child is designated as the fallthrough child
- the other child becomes an explicit jump target

ForestFire chooses the fallthrough child using training-time sample counts:

- the more common child is laid out immediately after the parent
- the less common child gets stored as the explicit jump

Why use sample counts here:

- they are already part of the semantic tree
- they are a reasonable proxy for the hot path during inference
- choosing the common child as fallthrough increases the chance that the common path avoids an extra jump

This is a runtime-only choice. The semantic tree still has left and right children exactly as before.

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

This matters most for `id3` and `c45`, where the semantic structure naturally wants to branch on discrete bins rather than collapsing everything into a binary threshold tree.

At the semantic level, a multiway node still says:

- feature `f`
- branch on observed bins
- use a fallback leaf if the observed bin does not match one of the trained branches

At the runtime level, the cost becomes:

- read one projected bin id
- bounds-check it against the lookup table
- either jump to the matching child or return the fallback leaf payload

That is substantially more regular than scanning a list of branches one by one.

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

In practical terms, an oblivious tree is close to:

- one feature index per depth
- one threshold per depth
- one bit contribution per depth
- one leaf array indexed by the accumulated bit pattern

That regularity is what makes the current SIMD path feasible there but not yet equally profitable for arbitrary CART-style trees. Arbitrary trees diverge in shape; oblivious trees do not.

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

Two details matter here:

### 1. Projection is computed from model semantics, not runtime guesses

ForestFire does not infer “used features” from observed inference traffic.

It computes them from the semantic model itself:

- scan all trees
- collect every feature index that appears in a split
- sort and deduplicate them

That means the projection is deterministic and exact with respect to the trained model.

### 2. Projection changes runtime feature indices, not the public schema

Suppose the semantic model uses features `[0, 4, 9]`.

The optimized runtime remaps them into local indices:

- semantic `0` -> runtime `0`
- semantic `4` -> runtime `1`
- semantic `9` -> runtime `2`

So runtime nodes refer to a dense local feature space, but callers still provide the original full feature set. This is important because it keeps the optimized model interoperable with the same user-facing API as the base model.

The user-facing metadata:

- `used_feature_indices`
- `used_feature_count`

exists so you can inspect that projected feature set directly.

## Batched preprocessing

Prediction cost is not just model traversal. It also includes turning user input into the internal binned representation.

ForestFire therefore preprocesses batches of rows together when possible.

Why:

- per-row conversion overhead dominates small or shallow-model workloads
- batching lets the runtime reuse feature metadata and conversion logic
- compact batch formats can then be fed directly into compiled runtimes

This is also why `polars.LazyFrame` prediction is batched in chunks instead of being materialized and traversed one row at a time.

The important implementation distinction is:

- single-row or tiny workloads prefer low overhead
- multi-row workloads benefit from a shared preprocessing step

ForestFire therefore keeps separate paths:

- a scalar row-oriented path
- a batch-oriented path for optimized runtimes

For `LazyFrame` specifically, the runtime does not try to collect the whole frame eagerly unless it is already small. Instead it slices the frame into chunks of about `10_000` rows, preprocesses each chunk, predicts it, and appends the results. That keeps memory use bounded while still letting the optimized runtime operate on batches.

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

Today the ordering heuristic is intentionally conservative:

- root or primary split feature first
- then used-feature count
- then the full used-feature set as a stable tiebreaker

The reason to keep it simple is that it needs to be:

- deterministic
- cheap to compute
- obviously semantics-preserving

This is not trying to be a learned scheduling policy. It is a locality hint that is easy to reason about and easy to snapshot in the compiled artifact.

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

This is closely tied to the current auto-binning strategy:

- bins are chosen as powers of two
- the count is capped
- each realized bin must contain at least two rows

That makes the realized bin domain small enough that compact integer storage is useful not just as a theoretical option but as the default fast path.

## Parallelism and SIMD

The optimized runtime currently uses CPU parallelism and SIMD where the structure supports it.

The main choices are:

- row-parallel chunking for large enough batches
- scalar fast paths for small workloads
- SIMD-oriented oblivious execution where the structure is regular enough

Why not apply the exact same SIMD strategy everywhere:

- arbitrary CART-style trees diverge quickly across rows
- oblivious trees keep the same split shape at every depth
- SIMD works best when many rows can execute the same instruction pattern

So the runtime is selective:

- exploit SIMD where the structure is regular
- exploit batching and locality where full SIMD is not yet worthwhile
- keep a cheap scalar path when the batch is too small to amortize setup costs

This matters because “optimized inference” is not one trick. It is the combination of:

- better layouts
- less preprocessing
- better memory density
- batching
- parallel chunking
- selective SIMD

## Compiled artifacts

The compiled artifact format exists for one reason: runtime lowering can itself be meaningful work.

If you repeatedly load the same optimized model, you do not want to:

- deserialize the semantic model
- recompute feature projection
- rebuild the lowered runtime layout
- reorder ensemble members again

every time.

So the compiled artifact stores:

- the semantic IR
- the projected feature set
- the lowered runtime layout

That gives you two useful guarantees:

- the semantic meaning is still preserved and inspectable
- reload can skip the expensive lowering step

This is why the compiled artifact is not a replacement for the IR. It is a cache of execution-oriented lowering built on top of the same semantic contract.

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

It also explains why optimized and non-optimized models deliberately export the same IR:

- optimization changes execution strategy
- it does not change the meaning of the model
- therefore the canonical semantic export should stay identical

That choice prevents a common failure mode in inference systems where the “fast path format” quietly becomes the only real model definition.

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

The most important practical takeaway is this:

- if your workload is “score one tiny row once”, the semantic model is often good enough
- if your workload is “score many rows repeatedly with the same model”, the optimized runtime has much more room to pay for itself

That is also why the library keeps both views instead of pretending one runtime strategy is always best.
