# Intermediate Representation

ForestFire has an explicit model IR instead of treating serialization as a direct dump of internal Rust structs.

That decision is central to the project.

The IR is the stable semantic layer between:

- training-time model structures
- optimized inference runtimes
- Python and Rust bindings
- serialized artifacts on disk

The important word there is semantic.

The IR is not trying to be the fastest possible execution format. It is trying to be the clearest possible statement of:

- what inputs the model expects
- what each split means
- what each leaf payload means
- how outputs should be interpreted
- which preprocessing assumptions must be preserved for inference to remain correct

## Why the IR exists

Many libraries effectively serialize whatever their internal training structs happen to look like.

That is easy at first, but it creates problems quickly:

- trainer internals leak into the exported format
- optimized runtimes become hard to reconstruct safely
- introspection and export drift apart
- changing internal layouts becomes risky

ForestFire avoids that by making the IR explicit.

The IR describes what a model *means*, not how a particular Rust struct happens to store it in memory.

That is why:

- `Model`
- `OptimizedModel`
- JSON serialization
- tree introspection
- dataframe export

can all agree with one another even though they do not store or consume the model in the same way internally.

## What the IR contains

The top-level package is `ModelPackageIr`.

It includes:

- producer metadata
- model structure
- input schema
- output schema
- inference options
- preprocessing description
- postprocessing description
- training metadata
- integrity / compatibility metadata

In practical terms, the IR answers questions like:

- what algorithm family produced this model?
- how many trees are there?
- how are the features binned?
- what do leaf values mean?
- how should outputs be interpreted?
- what assumptions does an inference runtime need to preserve?

It also answers a subtler but equally important question:

- what parts of the system are allowed to change without changing model meaning?

For ForestFire, that includes:

- optimized runtime layout
- feature projection used by optimized inference
- ensemble runtime ordering
- compiled artifact structure

None of those are part of the canonical IR because none of them change the semantic function the model computes.

## Why this matters for users

The IR is not just an implementation detail.

It is what makes these features line up with one another:

- `to_ir()` / JSON serialization
- `optimize_inference()`
- `tree_structure(...)`
- `tree_node(...)`, `tree_level(...)`, `tree_leaf(...)`
- `to_dataframe()`

Without a stable semantic representation, those would each need their own partially overlapping view of the model.

With the IR, they all derive from the same source of truth.

This is also why optimized and non-optimized models export the same IR. If they exported different semantic artifacts, optimization would become a semantic transformation instead of an execution transformation, which is exactly what the design is trying to avoid.

## Tree representations in the IR

ForestFire exports trees in one of two structural forms:

- `node_tree`
- `oblivious_levels`

### `node_tree`

This is used for standard trees such as:

- `id3`
- `c45`
- `cart`
- `randomized`

The IR stores:

- explicit nodes
- node ids
- split metadata
- child references
- leaf payloads
- node statistics

This representation is natural for trees whose structure is irregular and branch-specific.

That includes:

- ordinary CART-like binary trees
- randomized trees, which are structurally like CART but differ in how candidate splits are chosen
- `id3` and `c45`, whose learned structure may include multiway branches

### `oblivious_levels`

This is used for oblivious trees.

Instead of storing arbitrary node connectivity, the IR stores:

- one split per level
- leaf indexing rules
- leaf payloads
- per-level and per-leaf statistics

This matches the semantics of oblivious trees directly:

- every node at the same depth shares the same split
- leaf selection can be described as bit accumulation

That is both more compact and more faithful than pretending an oblivious tree is just an ordinary node graph.

It also makes the runtime/lowering boundary cleaner:

- the IR expresses the native semantics of an oblivious tree
- the optimized runtime can then choose scalar, SIMD, or batch-oriented execution without first having to reverse-engineer a generic node graph back into a level-wise form

## Leaf payloads

Leaves are represented semantically, not just numerically.

The IR distinguishes:

- regression values
- classification class indices and class values

Why both index and value?

Because the runtime often wants compact class-index-based execution, while the user-facing API still needs a stable class order and the original class labels.

The IR preserves both sides of that story.

## Input schema and preprocessing

ForestFire treats preprocessing as part of the model contract.

That means the IR stores:

- feature count and feature ordering
- feature logical types
- whether feature names are accepted
- numeric bin boundaries
- binary-feature handling

This matters because ForestFire models do not consume arbitrary floating-point inputs directly on the hot path. They consume the binned representation implied by training-time preprocessing.

If that preprocessing were not described explicitly, a deserialized model would not be self-sufficient.

That point is especially important now that:

- training uses adaptive power-of-two numeric binning
- optimized runtimes project to the subset of used features
- compact runtime batches store bin ids as `u8` or `u16`

All of those runtime optimizations depend on the semantic preprocessing contract being explicit and reconstructible.

### Categorical transforms in the IR

Categorical models now also serialize their categorical transform contract in
the IR.

That categorical block records:

- the raw input feature schema
- the categorical strategy in use
- the feature selection for categorical handling when explicitly configured
- the base smoothing parameter
- the fitted transform state needed to reproduce encoded inference

This matters because categorical strategies in ForestFire are currently
implemented as explicit transforms into numeric or binary feature space.

So for a categorical model, the IR is intentionally describing two related but
different spaces:

- the raw user-facing input space
- the encoded feature space consumed by the tree structure

For `dummy`, the IR preserves the learned indicator expansion.

For `target`, the IR preserves the fitted per-category target-derived mappings
and priors.

For `fisher`, the IR preserves the learned category ordering that maps raw
categories onto numeric ranks before split evaluation.

That means a deserialized categorical model can once again accept raw mixed
inputs directly rather than requiring the caller to rebuild the transform out
of band.

### Raw schema vs encoded feature space

This split is the most important categorical IR subtlety.

The tree definitions themselves still refer to encoded feature indices, because
that is the feature space the learner actually trained on.

But the top-level input contract for a categorical model is still the raw input
schema, because that is what callers are expected to provide at prediction
time.

So the IR is effectively saying:

- these are the raw inputs the user supplies
- this is the categorical transform applied first
- these are the encoded features the trees then evaluate

That is why categorical IR support required more than just flipping a boolean
flag. The transform had to become part of the semantic contract.

## Output schema and postprocessing

The IR also explains how to interpret outputs.

That includes:

- raw outputs
- final outputs
- class ordering when applicable
- postprocessing steps

This is especially important for classification and boosting, where the raw runtime quantity and the final user-facing quantity are not always the same conceptual object.

## Training metadata

The IR keeps training metadata alongside model structure.

That includes:

- algorithm
- task
- tree type
- criterion
- canaries
- tree count
- max depth
- min-sample controls
- `max_features`
- seed
- OOB settings and score
- boosting parameters such as learning rate and gradient sampling fractions

This metadata is not needed to *score* the model, but it is essential for:

- reproducibility
- inspection
- debugging
- binding-level property reflection

That is why model objects can expose those parameters as properties without having to reach back into the original training call.

## IR vs optimized runtime

The IR is not the same thing as the optimized runtime.

That distinction is important.

The IR is designed for:

- semantic clarity
- portability
- serialization
- introspection

The optimized runtime is designed for:

- compact execution layouts
- batched traversal
- SIMD-friendly access patterns

The runtime is lowered *from* the semantic model and IR-compatible structure. It is not the canonical serialized form.

In practice, that means:

- the IR contains the full semantic feature space
- the optimized runtime may use a projected feature space internally
- the IR contains semantic tree ordering
- the optimized runtime may reorder ensemble members for locality
- the IR contains semantic node/leaf meaning
- the optimized runtime may use fallthrough layouts, lookup tables, and compact batch representations

That separation keeps the project flexible:

- runtime layouts can improve without breaking serialized artifacts
- introspection can stay stable even if the runtime gets more specialized

For categorical models there is one extra distinction:

- the IR describes the raw categorical input contract plus the transform into
  encoded feature space
- the optimized runtime still executes on the encoded feature space after that
  transform has been applied

## IR vs compiled optimized artifacts

The compiled optimized artifact is a separate layer on top of the IR.

It exists because optimized lowering itself is real work. A compiled artifact can cache:

- the semantic IR
- the lowered runtime layout
- optimized-runtime metadata such as feature projection

Why not make that compiled artifact the main model format:

- it is backend-oriented rather than semantics-oriented
- it is harder to diff, inspect, and validate manually
- runtime layouts are more likely to evolve than semantic model meaning

So the project keeps a clean separation:

- IR for semantic truth
- compiled artifacts for faster reload of one particular optimized runtime

## Schema generation

The checked-in schema lives at:

- `crates/core/schema/forestfire-ir.schema.json`

It is generated from the Rust IR definitions using `schemars`.

That gives the project two useful guarantees:

- the schema stays aligned with the actual code
- changes to the IR surface are visible and testable

The schema test exists precisely so accidental IR drift does not go unnoticed.

That matters more as the runtime grows more sophisticated. The more execution-side optimization ForestFire adds, the more important it becomes that the semantic layer stay explicit and regression-tested.

## Why the IR is a design feature, not just a file format

The most important point is that the IR is not an afterthought.

It is what lets ForestFire keep these promises at the same time:

- train through one API
- inspect what was learned
- serialize the result
- lower it for faster inference
- preserve the same semantics across Rust and Python

Without the IR, those concerns would tend to fragment into separate incompatible representations.

With the IR, they stay aligned.
