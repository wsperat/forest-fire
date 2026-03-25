# Next steps

This page is for follow-up work that does not belong to a single feature page.

Some next steps are about training, some about runtime work, some about docs,
and some about benchmarking. The goal is to keep forward-looking notes in one
place instead of scattering them across otherwise stable reference pages.

## Gradient boosting parallelism

One of the clearest remaining training gaps is gradient-boosting parallelism.

Random forests parallelize naturally across trees, but boosting does not. Each
stage depends on the current ensemble state, so the outer training loop is
inherently sequential. The practical way to improve GBM CPU, and eventually GPU,
utilization is to make each individual tree fit much more parallel internally.

The highest-value strategy is:

- parallel histogram building with thread-local `count` / gradient / Hessian
  accumulators and a reduction step
- parallel split search across features once those histograms exist
- continued histogram subtraction so one child can be derived from the parent
  and its sibling instead of being rebuilt
- level-wise batching of active nodes where that is compatible with the tree
  family
- parallel row-index partitioning once the winning split has been chosen
- SIMD-friendly accumulation in the histogram hot path

The most important implementation detail is to separate stage-level seriality
from node-level parallelism:

- the boosting stage loop stays serial
- the work inside one stage should be aggressively parallel

That is the basic strategy used by systems like XGBoost, LightGBM, and
CatBoost. In practice, the first two steps above, histogram building and
feature-parallel split scoring, are likely to deliver the biggest visible gain.

## Random-forest training on wide data

Another clear next step is reducing RF training cost once feature counts become
moderate or large.

The current implementation parallelizes well across trees, but that is not the
same thing as making each tree cheap. In practice, the slowdown on wider tables
usually means the cost per node still scales too directly with the number of
features.

The most likely causes are:

- histogram construction is still too expensive in the hottest paths
- feature access is not cache-friendly enough on wide binned tables
- histogram subtraction and reuse are still incomplete in some important paths
- too much per-node scratch rebuilding or temporary allocation remains
- feature subsampling does not cut off enough work early enough

The main improvements to target are:

- make histogram construction the dominant optimization focus
- push histogram reuse further so one child can be built and the sibling derived
  by subtraction
- keep the training representation strongly feature-major so repeated scans over
  candidate columns stay cache-friendly
- batch active nodes by level where that improves locality and feature-parallel
  work
- ensure `max_features` gates the real hot path rather than only the final split
  comparison

This is one of the cases where profiling matters more than intuition. The most
useful breakdown to collect on wide RF workloads is:

- histogram building
- split scoring
- row partitioning
- temporary allocations
- bootstrap/index indirection overhead

The likely root issue is not lack of outer parallelism. It is that per-node
per-feature work is still too expensive once the width of the table grows.

## Split `crates/core/src/lib.rs` by responsibility

`crates/core/src/lib.rs` is getting large enough that it is starting to act as
a gravity well.

The abstractions are still understandable, but the file is carrying too many
jobs at once:

- model APIs
- inference-table construction
- optimized runtime definitions
- compiled artifact serialization
- batch inference logic
- SIMD helpers
- introspection glue

That is usually the point where future changes get harder than they need to be.

A good cleanup pass would split it into submodules along responsibility
boundaries, for example:

- `inference_input.rs`
- `optimized_runtime.rs`
- `compiled_artifact.rs`
- `introspection.rs`
- `model_api.rs`

The goal is not to change semantics. It is to reduce the amount of unrelated
context a contributor has to hold in their head before making routine changes.

## Consolidate tree-code duplication

The tree code is clearly in the transition zone between active optimization work
and needing a cleanup pass.

In particular, `classifier.rs` contains multiple layers of "fast" and
"fallback" split scoring, plus some older dead-code-marked paths that have not
yet been fully removed. That is not a correctness problem by itself, but it is
exactly the kind of layering that becomes hard to reason about once the codebase
grows further.

A worthwhile next step is a consolidation pass that:

- removes obsolete paths
- tightens the split-scoring hierarchy
- makes the fast path and fallback path boundaries explicit
- reduces the amount of near-duplicate scoring logic

## Stress-test randomization and seed mixing

The feature-subset and node-randomization seeding strategy deserves especially
careful validation.

Custom seed mixing is often necessary in ensemble code, but it is also one of
the easiest places for subtle bias or accidental correlation to creep in. Even
when the code is logically correct, weak seed derivation can reduce the
effective randomness of forests or randomized split search.

This area would benefit from more than ordinary unit tests:

- determinism checks across repeated runs
- property tests for seed stability and reproducibility
- tests that look for accidental correlations in feature-subset choices
- broader ensemble-behavior checks under many seeds

## Document semantic edge-case invariants

There are several API and semantics choices that appear intentional but are
surprising enough that they should probably be documented as explicit
invariants.

Examples:

- forests ignore canaries entirely
- boosting keeps canaries active
- classification labels are encoded as sorted numeric values
- class tie-breaking favors later indices in some `max_by` calls

These may all be valid design choices, but they affect behavior in edge cases.
Making them explicit would help users understand surprising outcomes and would
also make regression testing easier.

## Keep investing in runtime parity tests

The optimized runtime is one of the strongest parts of the system, but it is
also one of the highest-risk maintenance areas.

There are multiple representations of the same underlying model semantics:

- the semantic model path
- the optimized runtime path
- compiled serialization artifacts
- introspection parity requirements

That means there are multiple places where the same truth must remain aligned.
The existing tests already help a great deal, but this is exactly the area
where round-trip and parity tests are worth continuing to expand.

High-value coverage here includes:

- semantic model vs optimized runtime prediction parity
- optimized runtime vs compiled artifact parity
- serialization and reload round-trips
- introspection parity across semantic and optimized forms

## Split `classifier.rs` by concern

The classifier module is strong, but it is dense enough that more internal
decomposition would pay for itself.

Right now there are enough responsibilities in that file to justify splitting
it into smaller units such as:

- training orchestration
- split scoring
- histogram construction
- partitioning
- oblivious-specific logic
- IR conversion

This is partly a readability improvement, but it also helps make future
optimization work safer. Smaller files with clearer responsibility boundaries
make it easier to reason about which changes should affect behavior, which
should affect only performance, and which should be pure refactors.
