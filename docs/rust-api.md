# Rust API

The Rust side is split across crates:

- `forestfire-data`
- `forestfire-core`
- `forestfire-inference`

## Core training entrypoint

```rust
use forestfire_core::{train, TrainConfig};

let model = train(&table, TrainConfig::default())?;
```

## Important core types

- `TrainConfig`
- `TrainAlgorithm`
- `Task`
- `TreeType`
- `Criterion`
- `Model`
- `OptimizedModel`

## Data crate

The `forestfire-data` crate provides the training-table abstractions and preprocessing/binned storage used by the learners.

## Inference crate

The `forestfire-inference` crate contains inference-focused runtime utilities on top of the model IR and compiled runtimes.

## Publishing order

The crates should be published in dependency order:

1. `forestfire-data`
2. `forestfire-core`
3. `forestfire-inference`
