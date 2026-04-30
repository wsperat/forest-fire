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

A substantial portion of that plan is now in place:

- the second-order tree path has an explicit “evaluate one node” step separate
  from the recursive child-building step, which is the structural prerequisite
  for level-wise active-node batching
- standard second-order CART/randomized trees now use an active-node frontier,
  so a whole depth is evaluated before any node at that depth partitions the
  shared row-index buffer
- same-depth frontier evaluation runs in parallel across all nodes at a given
  depth and, simultaneously, each node's split scoring runs feature-parallel
  within the same rayon thread pool via nested work-stealing
- same-depth row partitioning runs in parallel across disjoint row-buffer
  slices after split choices have been fixed
- child histogram construction for the next frontier runs node-parallel across
  all splitting nodes, with each node's smaller child built feature-parallel
  and the larger child derived by subtraction from the parent histogram
- second-order histogram construction has a parallel-capable shared helper
  used across the histogram and split-scoring hot paths

The two-level approach — node-parallel outer loop with feature-parallel inner
operations, all sharing a single rayon thread pool via work-stealing — keeps
all available threads busy at every depth, including early depths where the
frontier is small.

The remaining concrete implementation steps are:

- add more aggressive SIMD work in histogram accumulation and reduction hot
  paths
- profile whether the gradient / hessian recomputation and prediction-update
  steps per stage are a meaningful fraction of wall time on large datasets, and
  parallelize those if so

## Experimental alternatives to second-order leaf optimization

Another worthwhile GBM track is broadening the leaf-weight optimizer beyond the
current second-order Newton approximation.

The immediate goal is not to replace the default Newton path. It is to make the
leaf-update step pluggable so ForestFire can experiment with higher-order Taylor
methods, Halley-style updates, and non-Taylor surrogates while keeping the
default behavior stable and well-tested.

The most important architectural rule is to keep two concerns separate:

- tree structure selection
- leaf-value optimization after the structure is fixed

That split keeps the current histogram-based split search usable while making it
possible to compare alternative leaf optimizers without rewriting the whole GBM
trainer.

### 1. Refactor the optimization interface

- extract leaf-weight optimization behind a shared interface
- keep the current second-order Newton step as the baseline implementation
- add explicit experimental optimizer implementations:
  - `SecondOrderOptimizer`
  - `ThirdOrderTaylorOptimizer`
  - `FourthOrderTaylorOptimizer`
  - `HalleyOptimizer`
  - `PadeOptimizer`
  - `LineSearchOptimizer`
  - `TrustRegionWrapper`
  - `AsymmetricQuadraticOptimizer`
  - `PiecewiseApproxOptimizer`
- make optimizer selection affect leaf fitting, not the default split-search
  path

### 2. Third-order Taylor approximation

- add a per-sample third-derivative API such as `d3loss(y, y_pred)`
- aggregate the per-leaf third-derivative statistic `S = sum(d3)`
- implement the per-leaf local objective
  - `G w + 0.5 (H + lambda) w^2 + (1/6) S w^3`
- solve the stationary condition as a quadratic in `w`
- handle roots conservatively:
  - keep only real roots
  - evaluate the leaf objective at each admissible root
  - prefer the root closest to the Newton step when solutions are otherwise
    competitive
  - reject very large-magnitude roots
- fall back to the second-order update when the cubic approximation is unstable
- log how often root ambiguity occurs

### 3. Fourth-order Taylor approximation

- add a per-sample fourth-derivative API such as `d4loss(y, y_pred)`
- aggregate the per-leaf fourth-derivative statistic `Q = sum(d4)`
- implement the per-leaf local objective
  - `G w + 0.5 (H + lambda) w^2 + (1/6) S w^3 + (1/24) Q w^4`
- solve the stationary condition as a cubic equation
- evaluate all real roots and apply stricter trust-region filtering than in the
  third-order case
- prefer the smallest-loss admissible solution
- clip more aggressively than with the third-order method
- fall back to second-order when the quartic surrogate is not trustworthy

### 4. Halley optimizer

- implement a Halley-style update from aggregated `(G, H, S)` statistics
- initialize from the Newton step `w0 = -G / (H + lambda)`
- run one deterministic Halley iteration rather than performing ambiguous
  root-selection logic
- add denominator safety checks
- apply step clipping or trust-region bounds
- fall back when:
  - the denominator is too small
  - the update becomes `NaN` or `Inf`
  - the candidate increases the true loss
- benchmark it against:
  - the current Newton update
  - the exact third-order Taylor root-selection path

### 5. Padé approximant

- implement a rational local surrogate rather than a pure Taylor polynomial
- start with a small numerator/denominator family such as quadratic over
  quadratic
- fit coefficients from local derivative information `(G, H, S, Q)` or a small
  local sampling procedure
- minimize the surrogate numerically per leaf
- guard against denominator singularities
- compare its stability to the third- and fourth-order Taylor paths

### 6. Global line search

- after the tree structure is fixed, optimize a global stage scale
  - `alpha = argmin sum_i l(y_i, y_pred + alpha * f_t(x_i))`
- implement:
  - backtracking line search
  - optional coarse grid search
- add stopping criteria based on realized loss decrease
- keep the line-search path composable with all leaf optimizers rather than
  treating it as a competing training mode

### 7. Shared trust-region wrapper

- add a generic wrapper that enforces `|w| <= delta`
- make it usable around every optimizer except pure line-search-only updates
- support adaptive `delta` expansion and shrinkage
- reject or clip steps outside the region
- track clipping and rejection statistics

### 8. Asymmetric quadratic approximation

- implement a bounded skew-aware surrogate that preserves a single optimum
- keep the shape close to a quadratic near the origin
- avoid a full cubic term when the main goal is asymmetry without multiple roots
- validate whether it improves behavior on asymmetric losses

### 9. Piecewise approximation

- define optimizer regimes from prediction scale or gradient magnitude:
  - near zero: quadratic
  - moderate: asymmetric
  - extreme: linear or clipped
- enforce continuity at regime boundaries
- make thresholds configurable
- add explicit transition tests

### 10. Split-scoring integration

- keep the default gain computation second-order at first
- only add higher-order split-gain experiments behind an explicit opt-in mode
- preserve a clear boundary between:
  - split search
  - leaf optimization
- add an `experimental_split_gain` flag rather than silently changing the
  default tree-structure criterion

### 11. Safety and fallback logic

- centralize fallback to second-order when:
  - required derivatives are missing
  - all candidate roots are complex
  - values become `NaN` or `Inf`
  - step magnitude becomes excessive
  - the true loss increases
- add debug counters per optimizer:
  - fallback reason
  - root ambiguity count
- add optional verbose per-leaf tracing for optimizer diagnostics

### 12. Tests

- add unit tests for:
  - derivative correctness
  - leaf objective evaluation
  - root solving
- validate derivatives numerically with finite differences
- cover edge cases such as:
  - near-zero Hessians
  - very large gradients
  - ambiguous roots
- add a regression test that proves the default second-order path is unchanged

### 13. Benchmarks

- evaluate at least:
  - squared-error regression
  - logistic binary classification
- compare:
  - training loss
  - validation loss
  - convergence speed
  - runtime
  - fallback frequency
  - update magnitudes
- include synthetic cases designed to trigger instability

### 14. Documentation

- document every optimizer with:
  - the local objective or update formula
  - required derivatives
  - main tradeoffs
- add a comparison table covering:
  - speed
  - stability
  - ambiguity
- mark every non-second-order method as experimental
- provide small usage examples once the optimizer-selection API stabilizes

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

- the tree-construction interface is now pluggable instead of hard-coded to
  greedy search
- the current public builders are:
  - `GreedyBuilder`
  - `LookaheadBuilder`
  - `BeamSearchBuilder`
- next builder families still worth exploring:
  - `RefinementBuilder`
  - `OptimalTreeBuilder`

### 15.1 Lookahead

- implemented:
  - shortlist the top `K` splits by immediate gain
  - re-score the shortlisted splits with bounded future expansion
  - expose configuration:
    - `lookahead_depth`
    - `lookahead_top_k`
    - `lookahead_weight`
- next work:
  - profile deeper horizons vs real quality gain
  - tune better defaults by algorithm and tree family
  - add stronger tracing and diagnostics for why a lookahead candidate wins

### 15.2 Beam search

- implemented:
  - a public `beam` builder with width `beam_width`
  - width-limited continuation search layered onto local split rescoring
  - the same public configuration family used by lookahead:
    - `lookahead_depth`
    - `lookahead_top_k`
    - `lookahead_weight`
    - `beam_width`
- next work:
  - move from local continuation search toward a fuller beam over partial-tree
    states
  - define a stronger partial-tree score from:
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
- measure whether stronger implemented builders such as lookahead and beam
  reduce the number of boosting rounds required to hit the same validation
  quality

## Constraint-aware modeling

Another high-value direction is making models stronger by making them more
structured rather than only more flexible.

Many real tabular problems come with known shape constraints:

- increasing income should not reduce approved credit under the intended policy
- higher dose should not decrease expected toxicity risk in some medical models
- some feature interactions are allowed while others should be forbidden

Encoding those priors can improve both generalization and trustworthiness. It
also changes the optimization problem in ways that are more interesting than
simply growing deeper trees.

The most useful roadmap items here are:

- add monotonic constraints for CART, forests, and boosting
- add interaction constraints so only approved feature groups may co-occur on a
  path or within one tree
- add optional fairness or policy-style constraints once the monotonic path is
  stable
- make constraint violations observable during training rather than silently
  clipped away

### Monotonic constraints

- allow per-feature monotonic directions:
  - increasing
  - decreasing
  - unconstrained
- filter split candidates and leaf updates that would violate the required
  ordering
- ensure boosted stage updates preserve the global monotonic contract
- benchmark the tradeoff between:
  - constrained training loss
  - validation quality
  - monotonicity violation rate under numerical edge cases

### Interaction constraints

- add configuration that limits which features may interact inside one path or
  one tree
- start with simple allow-list groups rather than a full constraint language
- integrate the constraint into:
  - split candidate generation
  - lookahead and beam-search builders
  - feature-importance and introspection output
- measure whether interaction constraints improve generalization on wide,
  weak-signal tables

### Constraint diagnostics

- add explicit reports for:
  - rejected splits due to constraints
  - constrained-vs-unconstrained objective gap
  - paths that sit near feasibility boundaries
- expose those diagnostics in introspection and dataframe export rather than
  keeping them trainer-internal

## Oblique and hybrid split families

The current tree families are all axis-aligned. That keeps training and runtime
simple, but it also limits the shape of functions that one split can express.

An important experimental next step is adding split families that can separate
examples using combinations of features rather than one threshold on one
feature.

This can improve model quality substantially when:

- useful boundaries are rotated rather than axis-aligned
- many weak numeric predictors only become strong together
- the user wants shallower, stronger trees instead of many tiny axis-aligned
  fragments

### Sparse oblique splits

The first sparse oblique implementation now exists.

Current state:

- the library supports an experimental pairwise oblique split type of the form:
  - `w_1 x_i + w_2 x_j <= t`
- oblique is available for:
  - `dt`
  - `rf`
  - `gbm`
  with `cart` and `randomized` tree types
- semantic IR and optimized-runtime support are in place
- missing-value handling is implemented per participating feature

The next oblique steps are now about extending and tuning that baseline:

- move beyond strictly pairwise splits when the added search cost is justified
- add stronger regularization and budgeting around when oblique nodes are
  allowed to appear
- benchmark whether the current “all pairs at the node” search should become
  adaptive again for larger feature spaces
- compare oblique-vs-axis tradeoffs at matched latency budgets

### Hybrid builders

- allow a tree builder to mix:
  - ordinary axis-aligned nodes
  - sparse oblique nodes only where they buy clear objective improvement
- add a budget such as:
  - maximum oblique nodes per tree
  - maximum features per oblique split
- preserve a clean fallback to pure axis-aligned training for portability and
  runtime simplicity

### Runtime and export implications

The semantic and optimized-runtime groundwork now exists, but runtime work is
still open.

The next runtime/export implications are:

- decide whether oblique optimized inference should stay row-wise or gain a
  batched projected-dot-product path
- benchmark whether a few strong oblique nodes outperform many ordinary nodes
  at equal prediction cost
- decide how far compiled-runtime specialization for oblique nodes should go
  before the added complexity stops paying for itself

## Richer leaf models

Another way to improve tree quality is to keep the structure tree-shaped while
making leaves more expressive than a constant.

This is especially attractive when a leaf contains a region that is mostly
simple but not truly flat. Instead of forcing the tree to keep splitting until
that region looks constant, it may be better to keep a coarser partition and
fit a richer local model in the leaf.

### Linear leaves and model trees

- add experimental linear leaves for regression first
- fit tiny regularized local models inside leaves using only the features seen
  on the path or a very small active subset
- compare:
  - one stronger tree with linear leaves
  - many deeper constant-leaf trees
- keep the optimized runtime path explicit, because linear leaves change the
  scoring contract

### Residualized leaves for boosting

- experiment with stage learners that still use tree structure but allow each
  leaf to fit:
  - a constant
  - a tiny linear correction
  - a calibrated one-dimensional response
- evaluate whether stronger leaves reduce the number of required boosting
  rounds enough to offset the extra per-leaf cost

### Safety and regularization

- require explicit regularization stronger than constant leaves
- log when richer leaves overfit tiny support regions
- add fallback to constant leaves when local conditioning is poor or the leaf
  support is too small

## Calibration and distributional outputs

Improving a model is not only about its point predictions. For many use cases,
better probabilities, intervals, and uncertainty estimates are the real model
improvement.

ForestFire already has good structure for semantic postprocessing, so this is a
natural place to push beyond ordinary point-estimate trees.

### Probability calibration

- add optional post-training calibration for classification:
  - Platt scaling
  - isotonic regression
  - temperature scaling for boosted margins where appropriate
- compare calibrated and uncalibrated:
  - log loss
  - Brier score
  - calibration curves
- decide whether calibrated wrappers should be:
  - separate artifacts
  - part of the semantic model contract

### Quantile and interval prediction

- add quantile-regression objectives and leaf semantics
- support prediction intervals from:
  - direct quantile models
  - conformal wrappers over existing models
- benchmark not just width but coverage accuracy and conditional coverage drift

### Distributional leaves

- experiment with leaves that predict a small parametric distribution rather
  than only a mean
- good early targets are:
  - Gaussian-style regression leaves
  - binary probability leaves with better calibrated uncertainty metadata
- extend the IR only after the semantic contract is clear and stable

### Evaluation

- add benchmark suites that score:
  - calibration error
  - interval coverage
  - sharpness vs coverage tradeoffs
  - uncertainty quality under covariate shift

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

## Extend categorical-feature support

ForestFire now implements three transform-based categorical strategies through
the unified `train(...)` interface: `dummy`, `target`, and `fisher`. Those
strategies are documented on the [Categorical Strategies](categorical-strategies.md)
page.

The current implementation uses explicit table-layer transforms rather than
native categorical split predicates. That is a practical tradeoff:

- it makes categorical support available through the existing numeric and binary
  split machinery
- but it means the learned trees operate on transformed feature space rather
  than on native category-membership tests

The remaining forward-looking work in this area is:

### Native categorical subset splits

The transform-based strategies do not support grouped category tests like:

```text
x in {"red", "green"} -> left
x in {"blue"} -> right
```

A single node can express that rule natively. The transform-based strategies
approximate it with either multiple binary columns (`dummy`) or a one-dimensional
ordered surrogate (`target`, `fisher`).

Native subset split support for CART-style learners would require:

- a first-class categorical feature representation in the training table
- per-node category histograms instead of only numeric-bin histograms
- split metadata that can express "left category set" rather than only
  threshold comparisons
- runtime support for fast category-membership tests
- IR support for categorical split predicates and category vocabularies

That is likely the strongest long-term end state for medium-cardinality
categoricals, but it is a larger architectural change than the transform-based
strategies.

### Leakage controls for target-style strategies

The current `target` and `fisher` strategies use smoothed training-set
statistics. That is a practical first implementation, but it does not fully
prevent target leakage.

A more rigorous approach would use out-of-fold or CatBoost-style prefix
statistics, so each row is encoded only from earlier rows rather than from the
full training set. That matters most for gradient boosting, where stage-wise
residual fitting is especially sensitive to weak spurious signal.

### Canary-informed smoothing

Canaries currently influence encoding strength for `target` and `fisher` by
increasing effective smoothing when shuffled-category signal looks too
competitive. See [Canary Strategy: Categorical variables](canaries.md#categorical-variables)
for how that works.

That mechanism is still heuristic. Stronger and more principled shrinkage rules,
especially for rare categories and high-cardinality columns in a boosting
setting, remain open work.
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
