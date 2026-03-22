# ForestFire

ForestFire is a tree-learning library with a Rust core and a Python API.

It is built around three ideas:

- one unified `train(...)` interface instead of learner-specific entrypoints
- one unified `Table` abstraction for training data
- one explicit model IR for serialization, portability, and runtime lowering

## What exists today

- decision trees, random forests, and gradient boosting
- classification and regression
- optimized inference runtimes
- model introspection and dataframe export
- Python and Rust APIs
- JSON model serialization

## Documentation map

- [Getting Started](getting-started.md): install and first training runs
- [Python API](python-api.md): Python surface and input handling
- [Rust API](rust-api.md): Rust crates and training entrypoints
- [Training](training.md): algorithms, parameters, and stopping behavior
- [Models And Introspection](models.md): prediction, optimization, serialization, and tree inspection
- [Benchmarks](benchmarks.md): benchmark tasks and artifact locations
- [Releasing](pypi-release.md): Python and Cargo release flows

## Project links

- Repository: [wsperat/forest-fire](https://github.com/wsperat/forest-fire)
- Python package: `forestfire-ml`
- Python import name: `forestfire`
