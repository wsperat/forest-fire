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

## Experimental tree-building strategies

Another useful training track is experimenting with stronger tree-construction
strategies than the current greedy builder.

This is distinct from leaf optimization. Higher-order or alternative leaf
optimizers change how a fixed tree assigns leaf values. The work here changes
how the tree structure itself is chosen.

The baseline should remain explicit:

- `GreedyBuilder` stays the default and the main speed baseline
- stronger builders should begin as clearly experimental alternatives

### 15. Refactor the tree-construction interface

- add a pluggable tree-builder interface so the current greedy search is not
  hard-coded as the only option
- keep the current implementation as `GreedyBuilder`
- reserve explicit experimental builders such as:
  - `LookaheadBuilder`
  - `BeamSearchBuilder`
  - `RefinementBuilder`
  - `OptimalTreeBuilder`

### 15.1 Lookahead

- implement depth-1 lookahead for split scoring
- for each node, shortlist the top `K` splits by immediate gain
- re-score the shortlisted splits by performing one-step child expansion
- add configuration:
  - `lookahead_depth`
  - `lookahead_top_k`
  - `lookahead_weight`

### 15.2 Beam search

- implement a beam over partial trees with width `beam_width`
- define a partial-tree score from:
  - current leaf objective
  - a heuristic estimate of future value
- deduplicate equivalent partial trees
- add budget controls:
  - maximum expansions
  - maximum memory
  - early stopping

### 15.3 Post-build non-greedy refinement

- build the tree greedily first
- revisit internal nodes one at a time while keeping topology fixed
- re-optimize split feature, threshold, and downstream routing
- recompute leaf weights after each accepted change
- stop after a fixed number of passes or when no more improvement is found

### 15.4 Optimal or near-optimal shallow search

- add a shallow-tree experimental mode using branch-and-bound or dynamic
  programming style search
- restrict the first version to:
  - small depth
  - small feature subsets
- add a time-budgeted anytime mode
- compare directly against greedy and lookahead builders

### 15.5 Evaluation

- benchmark tree builders on:
  - training objective
  - validation objective
  - wall-clock time
  - tree size and realized depth
  - number of accepted refinements
- measure whether stronger trees reduce the number of boosting rounds required
  to hit the same validation quality

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

## Revisit canary policy for boosting

The current canary mechanism is useful because it gives ForestFire an explicit
"stop when the best real split is indistinguishable from shuffled noise"
criterion.

That is a good fit for some settings, but boosting is a special case. In
boosting, later trees are not trying to explain the full target from scratch.
They are trying to explain the *residual margin* left behind by the existing
ensemble. That means a stage can be locally weak in an absolute sense while
still being globally useful once shrinkage and many stages are taken into
account.

The `make_moons` benchmark exposed exactly that failure mode:

- the CART binary-GBM path produced a zero-tree ensemble
- the model therefore stayed at the base score
- the resulting probability surface was flat at `0.5`

That outcome strongly suggests the current root-canary policy is too eager in
at least some second-order boosting settings.

The next step should not be to remove canaries from boosting entirely. It
should be to make the policy stage-aware and objective-aware.

The most promising directions are:

- compare the best real split against canaries using *margin reduction* or
  stage-loss reduction instead of a more generic tree-training signal
- require more evidence before stopping the first few boosting stages, where a
  weak but real root split may still unlock a useful ensemble trajectory
- distinguish "no signal at all" from "signal is weak but consistently better
  than the current ensemble baseline"
- allow boosting to continue when the best real split beats the unsplit leaf by
  a meaningful amount, even if it does not beat every canary statistic under
  the current rule
- consider a softer canary policy for boosting than for plain decision trees,
  random forests, or mean-target baselines

The practical goal is:

- keep canaries as a guardrail against fitting pure noise
- but avoid collapsing useful GBM runs into zero-tree ensembles on genuinely
  structured datasets

This should be treated as both an algorithmic and benchmarking task:

- add focused tests for binary-GBM datasets that should produce at least one
  tree
- use `make_moons` and similar low-dimensional nonlinear datasets as regression
  benchmarks for canary behavior
- record whether canary-triggered early stopping produced a zero-tree ensemble,
  because that is a distinct failure mode worth surfacing directly

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
