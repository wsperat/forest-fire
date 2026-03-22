# Intermediate Representation

ForestFire has an explicit model IR instead of treating serialization as a direct dump of internal Rust structs.

That decision is central to the project.

The IR is the stable semantic layer between:

- training-time model structures
- optimized inference runtimes
- Python and Rust bindings
- serialized artifacts on disk

## Why the IR exists

Many libraries effectively serialize whatever their internal training structs happen to look like.

That is easy at first, but it creates problems quickly:

- trainer internals leak into the exported format
- optimized runtimes become hard to reconstruct safely
- introspection and export drift apart
- changing internal layouts becomes risky

ForestFire avoids that by making the IR explicit.

The IR describes what a model *means*, not how a particular Rust struct happens to store it in memory.

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

That separation keeps the project flexible:

- runtime layouts can improve without breaking serialized artifacts
- introspection can stay stable even if the runtime gets more specialized

## Schema generation

The checked-in schema lives at:

- `crates/core/schema/forestfire-ir.schema.json`

It is generated from the Rust IR definitions using `schemars`.

That gives the project two useful guarantees:

- the schema stays aligned with the actual code
- changes to the IR surface are visible and testable

The schema test exists precisely so accidental IR drift does not go unnoticed.

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
