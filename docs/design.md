# Design And Architecture

## Why ForestFire is organized the way it is

ForestFire is intentionally built around a small number of strong abstractions instead of a large catalog of learner-specific entrypoints.

The core design choices are:

- one unified training surface
- one shared training-table abstraction
- one explicit semantic model representation
- one separate optimized runtime view

Those choices are there to solve real coordination problems in tree libraries.

## Unified training surface

The project uses `train(...)` as the main entrypoint rather than exposing a separate class or constructor for every learner family.

Why:

- users think in terms of task, tree family, and constraints before they think in terms of API object hierarchies
- the public surface stays stable as new learner families are added
- the Python and Rust layers can stay aligned instead of drifting into unrelated APIs

Tradeoff:

- the configuration object gets richer over time
- validation becomes more important because not every parameter applies to every algorithm

That tradeoff is accepted deliberately. The project prefers one explicit configuration matrix over many partially overlapping public entrypoints.

## Shared `Table` abstraction

The learners do not operate directly on arbitrary user containers. They operate on a common `TableAccess` interface backed by dense or sparse table implementations.

Why:

- preprocessing should happen once, not be duplicated inside every learner
- sparse-vs-dense storage is an execution detail, not a modeling concept
- canary generation, binning, and feature typing belong in one place if model semantics are going to stay coherent

Impact:

- every learner sees the same binned representation
- forests do not rebucket data per tree
- optimized inference can reuse the same feature-preprocessing semantics that training used

The table abstraction is one of the project’s highest-leverage design decisions because it keeps “what the data means” separate from “which learner is using it”.

## Shared tree internals

The project also deliberately shares as much tree-building machinery as it can
across classification, regression, and second-order boosting trees.

That shared layer now covers the nontrivial mechanics that are easy to let drift
apart if every learner owns them independently:

- histogram construction and subtraction
- in-place binary row partitioning
- feature-subset sampling
- seed derivation for node-local randomization
- randomized threshold selection

Why that matters:

- classification, regression, and GBM trees are different at the level of split
  objective and leaf payload
- they are not different at the level of “how do we build a histogram over the
  binned table?” or “how do we derive a sibling histogram from the parent?”

Keeping those mechanics shared reduces maintenance risk in two directions:

- performance fixes land once instead of being copied across three code paths
- randomization semantics stay aligned across ordinary trees, forests, and
  second-order stage learners

This is one of the cases where architectural cleanliness is also a correctness
and performance win. Shared internals make it much harder for one learner family
to accidentally diverge in subtle ways from the others.

The same pattern now shows up on the inference side as well:

- semantic feature usage is computed once from the model
- optimized runtimes reuse one projection-aware preprocessing path
- compiled artifacts snapshot that lowered runtime instead of inventing a second execution semantics

## Why binning is central

ForestFire is built around bounded numeric bins rather than exact threshold handling everywhere.

Why:

- repeated exact threshold rescans are expensive
- a bounded discrete search space enables histogram-based split search
- the same discretization can be reused by training, serialization, and optimized runtime lowering

Tradeoff:

- very fine-grained exact threshold behavior is approximated by bins
- the binning strategy becomes part of model semantics

That is acceptable because the project values regularity and portability more than preserving every raw numeric distinction internally.

The newer adaptive binning rules reinforce that design:

- bin counts stay powers of two
- they are capped
- auto binning keeps at least two rows in each realized bin

That choice is not only about training. It is also about the rest of the system:

- the IR can describe the preprocessing compactly
- optimized runtimes can use narrow integer storage
- runtime lookup tables become easier to size tightly

## Why canaries exist

Canaries are shuffled copies of already-preprocessed features that compete with real features during split search.

Why:

- impurity improvement alone does not tell you whether the model is still learning structure or just fitting noise
- a canary feature is a practical training-time baseline for “what if this split quality were random?”
- this makes stopping part of the split-selection process instead of a later clean-up stage

This is a strong design opinion:

- ForestFire prefers in-training noise competition
- it does not treat pruning as the primary answer to overgrowth

The other important design choice around stochastic training is that
randomization is deterministic but explicitly derived from stable training
context:

- ensemble-level seeds are mixed per tree or boosting stage
- node-local seeds are derived from the base seed, depth, salt, and the row set
  currently owned by the node
- randomized threshold selection uses the same deterministic context

That gives the library two properties at once:

- repeated runs with the same seed are reproducible
- different stages, trees, and nodes do not accidentally collapse onto the same
  pseudo-random choices

The implementation also now avoids depending on incidental row-buffer order
inside a node. The row set matters; the temporary ordering of that set does not.

That is also why canary behavior differs by algorithm:

- single trees use them directly as a local stopping signal
- boosting keeps them because late-stage residual fitting is especially prone to noise chasing
- random forests ignore them because bagging and feature subsampling are already the dominant regularizers there

## Why optimized inference is a separate model view

Training structures are rich in information:

- impurity
- gain
- sample counts
- multiway branch metadata
- class counts

Those are useful for debugging, inspection, and IR export, but they are not free on the hot scoring path.

So ForestFire treats runtime optimization as a lowering step:

- the semantic model stays the same
- the execution layout changes

This separation keeps two important properties at once:

- introspection still sees the full trained structure
- prediction can use a layout that is much closer to what CPUs want

That runtime lowering now includes several distinct transformations:

- removing training-only node payload from hot-path layouts
- remapping features into a compact projected space
- reordering ensemble members for better locality
- choosing specialized execution formats such as fallthrough binary layouts or oblivious level arrays

The important architectural point is that all of those are execution choices, not semantic changes.

## Why the IR is first-class

The IR is not an export afterthought. It is the semantic bridge between:

- training
- optimized inference
- serialization
- introspection

Why it matters:

- without an explicit semantic layer, optimized runtimes drift from trainer semantics
- without recorded preprocessing assumptions, “deserialize and predict” is not reproducible
- without a stable structural representation, introspection becomes tied to implementation details

The IR forces the project to answer explicitly:

- what a node means
- what a leaf payload means
- how preprocessing is represented
- which runtime transformations preserve semantics

That discipline is one of the reasons ForestFire can expose runtime lowering, dataframe export, and serialization without each feature inventing its own hidden interpretation of the model.

It also explains why optimized and non-optimized models intentionally share the same IR export:

- both are views of the same learned object
- optimization is allowed to change layout, not meaning
- the IR must therefore stay above runtime-specific details like projected local feature indices or ensemble execution order

That same reasoning is why the compiled optimized artifact is layered on top of the semantic model instead of replacing it.

## Why multiple tree families remain exposed

The project does not collapse everything into a single default binary tree family because the structural choice genuinely changes:

- inductive bias
- interpretability
- runtime shape
- suitability for ensembles

Examples:

- `id3` and `c45` are attractive when you want direct per-bin branching structure
- `cart` is the most general-purpose backbone for forests and boosting
- `randomized` is valuable when deliberate stochasticity is part of the learner
- `oblivious` gives up flexibility to gain regularity, which makes optimized execution and symmetric-tree reasoning much cleaner

## The recurring theme

Across the codebase, the same preference keeps showing up:

- make semantics explicit
- separate meaning from execution strategy
- prefer regular internal representations when they unlock system-wide gains
- let training-time design choices feed directly into runtime, export, and introspection

The new optimized-runtime work is a good example of that principle in practice:

- training determines which features and bins actually matter
- the semantic model records that meaning
- the optimized runtime exploits it through projection, compact batch storage, and locality-aware lowering
- the compiled artifact snapshots that result for fast reload

The throughline is still the same: meaning first, execution second, but execution is taken seriously enough to deserve its own explicit design.

That is the architectural throughline of the project.
