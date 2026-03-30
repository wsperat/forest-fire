# Rust API

The Rust side is split across crates:

- `forestfire-data`
- `forestfire-core`
- `forestfire-inference`

## Core training entrypoint

```rust
use forestfire_core::{train, TrainConfig};

let model = train(&table, TrainConfig::default())?;
let optimized = model.optimize_inference(Some(1))?;
```

The intended Rust lifecycle is:

1. build a training table through `forestfire-data`
2. train a semantic `Model`
3. use the semantic model for introspection and canonical serialization
4. derive an `OptimizedModel` when prediction speed matters
5. optionally snapshot the optimized runtime as a compiled artifact

## Important core types

- `TrainConfig`
- `TrainAlgorithm`
- `Task`
- `TreeType`
- `Criterion`
- `Model`
- `OptimizedModel`

## Core capabilities

`forestfire-core` currently provides:

- unified training dispatch
- decision trees, random forests, and gradient boosting
- optimized inference runtimes
- JSON IR serialization and deserialization
- tree introspection metadata
- compiled optimized runtime artifacts
- used-feature introspection for semantic and optimized models

Useful runtime-oriented methods:

- `Model::used_feature_indices()`
- `Model::used_feature_count()`
- `Model::optimize_inference(...)`
- `OptimizedModel::used_feature_indices()`
- `OptimizedModel::used_feature_count()`
- `OptimizedModel::serialize_compiled()`
- `OptimizedModel::deserialize_compiled(...)`

Optimized models still accept the full semantic feature space on input, but they lower the runtime into a compact projected feature space internally so batch preprocessing only touches the columns that appear in splits.

That means there are really three layers to keep in mind:

- `Model`: semantic meaning
- `OptimizedModel`: lowered runtime
- compiled artifact: serialized lowered runtime plus semantic IR

### Example: semantic model vs optimized runtime

```rust
use forestfire_core::{train, TrainConfig};
use forestfire_data::Table;

let table = Table::new(
    vec![
        vec![0.0, 0.0, 10.0],
        vec![0.0, 1.0, 10.0],
        vec![1.0, 0.0, 10.0],
        vec![1.0, 1.0, 10.0],
    ],
    vec![0.0, 0.0, 0.0, 1.0],
)?;

let model = train(&table, TrainConfig::default())?;
let optimized = model.optimize_inference(Some(1))?;

println!("{:?}", model.used_feature_indices());
println!("{:?}", optimized.used_feature_indices());
```

Those used-feature methods reflect the semantic split structure, and the optimized runtime uses them to project inference input before scoring.

### Example: compiled optimized artifact

```rust
let optimized = model.optimize_inference(Some(1))?;
let bytes = optimized.serialize_compiled()?;
let restored = forestfire_core::OptimizedModel::deserialize_compiled(&bytes, Some(1))?;
```

Use this when you want to preserve the lowered runtime layout across reloads instead of recomputing it from the semantic model each time.

## Data crate

The `forestfire-data` crate provides the training-table abstractions and preprocessing/binned storage used by the learners.

Important point:

- `Table` is the training-side abstraction
- inference can use raw rows, named columns, sparse binary columns, and `polars` data directly through `forestfire-core`

That mirrors the Python surface: training normalizes through tables, while prediction accepts user-facing inference inputs directly.

## Inference crate

The `forestfire-inference` crate contains inference-focused runtime utilities on top of the model IR and compiled runtimes.

## Rust usage notes

Use the core and data crates directly from the workspace today. The library is still early-stage, so the repository state should be treated as the source of truth for the public surface.

## Publishing order

The crates should be published in dependency order:

1. `forestfire-data`
2. `forestfire-core`
3. `forestfire-inference`
