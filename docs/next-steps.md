# Next steps

This page is for follow-up work that does not belong to a single feature page.

Some next steps are about training, some about runtime work, some about docs,
and some about benchmarking. The goal is to keep forward-looking notes in one
place instead of scattering them across otherwise stable reference pages.

Before looking at the remaining roadmap items, it is worth calling out cleanup
work that has already landed and should no longer be read as pending:

- `crates/core/src/lib.rs` is now split by responsibility, with training,
  introspection, forest, boosting, model API, and root test coverage living in
  dedicated modules instead of accumulating in the crate root
- `classifier.rs` is split by concern, with histogram handling, split scoring,
  partitioning, oblivious training, IR support, and tests factored into focused
  submodules
- shared histogram / partitioning / randomization internals are consolidated
  across classifier, regressor, and second-order tree builders via common
  helpers instead of duplicated implementations
- seed mixing and randomized selection behavior now have explicit invariants and
  repeatability coverage, so deterministic training behavior is tested rather
  than assumed

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
