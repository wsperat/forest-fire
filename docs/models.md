# Models And Introspection

## Prediction

- `predict(...)` returns hard predictions
- `predict_proba(...)` returns class probabilities for classification models

Classification prediction returns hard labels, while `predict_proba(...)` exposes the underlying class distributions when supported by the model.

That distinction is important because ForestFire does not treat “probability prediction” as an afterthought. For ensembles, especially forests and boosted classifiers, the probability path is the more semantically faithful one:

- forests aggregate per-tree probabilities and then derive hard labels
- optimized runtimes preserve those leaf distributions rather than collapsing everything to winner-take-all labels
- introspection and dataframe export expose the same learned leaf payloads that drive both `predict(...)` and `predict_proba(...)`

## Optimized inference

`optimize_inference(...)` lowers a trained model into a runtime-oriented representation for faster prediction.

It preserves model semantics while changing the execution layout.

That includes missing-value semantics: optimized models keep the learned
missing routing for the features you ask them to preserve, and otherwise retain
the canonical semantic model for serialization and inspection.

This split between training representation and runtime representation is one of the core design decisions in the project.

Why do it this way:

- the best structure for training is not automatically the best structure for scoring
- training nodes carry bookkeeping that is useful for inspection and serialization but wasteful on the hot prediction path
- runtime lowering lets the project keep one semantic model while still specializing execution for batch prediction
- optimized runtimes can compact the active feature space without changing what inputs the semantic model expects

The important design rule is:

- `Model` is the canonical semantic object
- `OptimizedModel` is a derived execution object

That distinction is what lets ForestFire keep introspection, serialization, and optimized inference aligned instead of forcing them to compete.

### What it changes internally

- prediction-only node layouts drop training-only fields
- compiled CART-style trees use a fallthrough layout
- multiway classifier nodes use dense bin lookup tables
- oblivious trees become compact level arrays plus a leaf table
- optimized runtimes project inputs down to the union of features the model actually uses
- forests and boosted ensembles reorder trees by simple feature-locality keys before lowering
- batch preprocessing happens ahead of traversal
- compiled runtimes use compact `u8`/`u16` column-major batch matrices
- `polars.LazyFrame` inputs are collected and scored in batches of about `10_000` rows

Why those changes help:

- fallthrough layouts reduce branch-heavy pointer chasing in binary trees
- dense lookup tables replace repeated branch scans in multiway nodes
- oblivious trees are naturally amenable to regular array-based execution
- feature projection avoids materializing and binning columns that are never touched by the trained model
- locality-oriented tree ordering gives ensembles a better chance of reusing hot feature columns and top-level metadata
- compact batch matrices reduce bandwidth and improve cache density
- batched preprocessing amortizes input conversion over many rows instead of repeating it per traversal

Another way to say this is that optimized inference changes three things at once:

1. which data structures represent the model
2. which feature columns are materialized at prediction time
3. which execution strategy is used for row batches

The optimized model is therefore not just “the same predictor with a faster tree walk”. It is a full runtime lowering pass.

### Feature projection

Optimized models still accept the full semantic input schema, but they no longer preprocess every feature eagerly.

Instead, ForestFire:

- inspects the semantic model
- computes the sorted union of all feature indices that appear in splits
- remaps the optimized runtime into that compact feature space
- preprocesses only those projected columns during optimized inference

This matters most when:

- upstream pipelines emit wide tables but each tree only touches a small subset
- forests or boosted ensembles repeatedly reuse a narrow set of strong predictors
- batch preprocessing cost is large enough to matter, not just traversal cost

Both `Model` and `OptimizedModel` expose:

- `used_feature_indices()`
- `used_feature_count()`

That makes it easy to inspect whether a trained model is genuinely sparse in feature usage before relying on the optimized path.

This is especially useful for ensembles, because the semantic feature count and the effective runtime feature count can differ substantially:

- the semantic model may have been trained on a wide table
- each tree may only use a small subset
- the optimized ensemble can then project to the union of all actually-used features

That is one of the main reasons optimized inference can improve not just traversal speed but total end-to-end scoring cost.

### Compiled optimized models

`OptimizedModel` can also be serialized into a compiled artifact.

That artifact keeps:

- the semantic IR
- the lowered runtime layout
- the feature projection metadata

The reason to keep all three is that they solve different problems:

- the semantic IR preserves meaning
- the lowered runtime preserves load-time work
- the projection metadata preserves how runtime-local feature indices map back to the semantic schema

So a compiled optimized model is best understood as a deployment cache for the optimized runtime, not as a new canonical model definition.

For categorical models, the compiled artifact also carries the categorical
transform metadata needed to convert raw mixed inputs into the encoded feature
space expected by the lowered runtime.

Where the impact is largest:

- large batches
- deep or ensemble-heavy models
- repeated scoring of already-trained models
- workflows where the same model serves many predictions and the one-time lowering cost is amortized

## Serialization

Available export paths:

- JSON model serialization
- JSON IR export
- compiled optimized runtime serialization

Compiled optimized artifacts retain both:

- the semantic IR
- the runtime-specific feature projection and lowered execution layout

That means reloading a compiled optimized model skips the lowering step without changing the model’s semantic serialization.

In other words, ForestFire deliberately has two layers of artifacts:

- canonical semantic artifacts for portability and inspection
- optional compiled runtime artifacts for faster load and predict paths

That split is what keeps the project flexible as runtime layouts evolve.

### IR

The IR exists to make inference semantics explicit and portable.

It includes:

- algorithm, task, tree type, and criterion
- explicit `node_tree` and `oblivious_levels` representations
- training-time numeric bin boundaries
- categorical transform metadata when categorical strategies were used
- leaf payloads for classification and regression
- node and leaf stats like sample counts, impurity, gain, class counts, and variance when relevant

The rationale for the IR is broader than export alone:

- it is the semantic contract between training and optimized inference
- it keeps serialization honest by forcing preprocessing and leaf semantics to be represented explicitly
- it makes introspection possible without inventing a second ad hoc debug format

In other words, the IR is not a side artifact. It is the shared meaning layer for the project.

That is why `Model` and `OptimizedModel` export the same IR JSON:

- both objects represent the same learned function
- only one of them stores it in a runtime-specialized way
- the IR must therefore describe the common semantics, not the optimization strategy

For categorical models, that common semantic layer now includes:

- the raw input schema
- the categorical strategy configuration
- the serialized transform state needed to reproduce encoded inference before
  tree evaluation

## Tree introspection

All tree-backed models expose:

- tree structure summaries
- node/level/leaf inspection
- prediction-value statistics
- dataframe export

Typical use cases:

- understanding realized tree size and depth
- inspecting learned splits and leaf payloads
- comparing optimized and non-optimized views
- inspecting one tree at a time inside forests and boosted ensembles

The introspection API exists because “tree model” users often need to answer questions that pure prediction APIs cannot:

- did the learner actually grow the shape I expected?
- which feature did a particular node split on?
- how deep did the ensemble trees become in practice?
- are the leaf values concentrated or spread out?

The design goal here is to expose the trained structure without making users decode the raw IR by hand.

There is also a practical runtime reason to keep introspection semantic:

- optimized runtimes may reorder trees or remap feature indices internally
- users generally want to inspect the trained meaning, not the lowered execution cache

So introspection stays anchored to the semantic model even when optimized inference is available.

## `to_dataframe(...)`

`to_dataframe(...)` is a tabular export of the tree structure:

- standard trees include split rows, leaf rows, and unmatched fallback leaves where relevant
- oblivious trees include one row per level and one row per leaf
- forests and boosted ensembles include `tree_index`

This method exists for interoperability more than aesthetics. A dataframe-like export is the easiest way to:

- join tree structure with downstream analysis code
- compare models programmatically
- inspect many trees at once in forests and boosting
- build custom reports without re-walking nested JSON structures
