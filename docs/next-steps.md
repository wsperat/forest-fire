# Next steps

This page is for follow-up work that does not belong to a single feature page.

Some next steps are about training, some about runtime work, some about docs,
and some about benchmarking. The goal is to keep forward-looking notes in one
place instead of scattering them across otherwise stable reference pages.

Recent cleanup work already completed and therefore intentionally not listed as
open items here includes:

- splitting `crates/core/src/lib.rs` by responsibility
- splitting `classifier.rs` by concern
- consolidating shared histogram / partitioning / randomization internals across
  classifier, regressor, and second-order tree builders
- hardening and stress-testing seed mixing and randomized selection behavior

## Gradient boosting parallelism

One of the clearest remaining training gaps is gradient-boosting parallelism.

Random forests parallelize naturally across trees, but boosting does not. Each
stage depends on the current ensemble state, so the outer training loop is
inherently sequential. The practical way to improve GBM CPU, and eventually GPU,
utilization is to make each individual tree fit much more parallel internally.

The implementation plan should be staged rather than treated as one large
rewrite:

1. Keep the boosting stage loop serial, but make one second-order tree fit more
   parallel internally.
2. Separate node evaluation from row-buffer mutation so active nodes can be
   evaluated in batches at one depth.
3. Make histogram construction and split scoring parallel across features and
   then across batches of active nodes.
4. Only after those pieces are stable, parallelize row partitioning and add
   more aggressive SIMD work in the histogram hot path.

The highest-value technical milestones are:

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

Some groundwork for that plan is now in place:

- the second-order tree path has an explicit “evaluate one node” step separate
  from the recursive child-building step, which is the structural prerequisite
  for level-wise active-node batching
- second-order histogram construction now has a parallel-capable shared helper
  instead of forcing the GBM path through a purely sequential per-feature build

The next concrete implementation step should be introducing an active-node
frontier for standard second-order trees so a whole level can be scored before
any node at that level mutates the shared row-index buffer.

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
