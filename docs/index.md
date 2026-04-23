# ForestFire

![ForestFire](forest-fire.jpg)

ForestFire is a tree-learning library with a Rust core and a Python API.

It is built around three ideas:

- one unified `train(...)` interface instead of learner-specific entrypoints
- one unified `Table` abstraction for training data
- one explicit model IR for serialization, portability, and runtime lowering

Those choices are deliberate:

- the unified training surface keeps the public API stable even as the library grows from single trees into forests and boosting
- the `Table` abstraction centralizes preprocessing, binning, sparse handling, and canary generation so the learners do not each re-implement data plumbing
- the explicit IR separates model semantics from trainer internals, which is what makes optimized inference, introspection, and serialization all line up with the same underlying meaning

## What exists today

- decision trees, random forests, and gradient boosting
- classification and regression
- optimized inference runtimes
- model introspection and dataframe export
- Python and Rust APIs
- JSON model serialization

ForestFire is intentionally opinionated in a few places:

- it prefers a small number of strong public concepts over many learner-specific entrypoints
- it treats training-time preprocessing as part of the model contract, not an invisible side effect
- it uses canaries as an in-training stopping signal instead of relying on post-hoc pruning
- it exposes optimized inference as a separate runtime view rather than pretending the training structure is automatically the best scoring structure

## Documentation map

- [Getting Started](getting-started.md): install and first training runs
- [Design And Architecture](design.md): the core abstractions and why they exist
- [Canary Strategy](canaries.md): why canaries exist, what they replace, and how they differ across DT, RF, and GBM
- [Runtime And IR](runtime.md): inference lowering, serialization, and execution design
- [Intermediate Representation](ir.md): the semantic model package, schema, and portability story
- [Python API](python-api.md): Python surface and input handling
- [Rust API](rust-api.md): Rust crates and training entrypoints
- [Examples](examples.md): end-to-end workflows from training through reload and batch scoring
- [Training](training.md): algorithms, parameters, and stopping behavior
- Builders:
  - [Lookahead Builder](lookahead-builder.md): shortlist-based future-aware split ranking
  - [Beam Builder](beam-builder.md): width-limited continuation search for split ranking
  - [Optimal Builder](optimal-builder.md): exhaustive subtree search with canary-driven stopping
- [Categorical Strategies](categorical-strategies.md): `dummy`, `target`, and `fisher` categorical handling through the native training API
- [Oblique Splits](oblique-splits.md): pairwise linear splits, weight computation, candidate competition, and when to use them
- [Models And Introspection](models.md): prediction, optimization, serialization, and tree inspection
- [Benchmarks](benchmarks.md): benchmark tasks and artifact locations
- [Next Steps](next-steps.md): forward-looking implementation and optimization notes
- [Releasing](pypi-release.md): Python and Cargo release flows

## Project links

- Repository: [wsperat/forest-fire](https://github.com/wsperat/forest-fire)
- Python package: `forestfire-ml`
- Python import name: `forestfire`
