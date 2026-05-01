# Next steps

This page is for follow-up work that does not belong to a single feature page.

Some next steps are about training, some about runtime work, some about docs,
and some about benchmarking. The goal is to keep forward-looking notes in one
place instead of scattering them across otherwise stable reference pages.

The priorities below are ordered from highest to lowest expected value for the
project's near-term usefulness and technical leverage.

## Priority order

1. **Revisit canary policy for boosting**
2. **Extend categorical-feature support**
3. **Experimental tree-building strategies**
4. **Random-forest training on wide data**
5. **Keep investing in runtime parity tests**
6. **Gradient boosting parallelism**
7. **Constraint-aware modeling**
8. **Oblique and hybrid split families**
9. **Calibration and distributional outputs**
10. **Richer leaf models and soft trees**
11. **Document semantic edge-case invariants**
12. **Experimental alternatives to second-order leaf optimization**

The detailed sections below remain grouped by topic, but this list is the
intended execution order.

## Gradient boosting parallelism

The remaining GBM parallelism work is now mostly hot-path tuning rather than
structural scheduler changes.

- SIMD-friendly accumulation in the histogram hot path
- Profile whether gradient / hessian recomputation and prediction-update steps
  per boosting stage are a meaningful fraction of wall time on large datasets,
  and parallelize those if so

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
strategies.

This is distinct from leaf optimization. Higher-order or alternative leaf
optimizers change how a fixed tree assigns leaf values. The work here changes
how the tree structure itself is chosen.

### 15.1 Lookahead and beam follow-up

- profile deeper horizons vs real quality gain
- tune better defaults by algorithm and tree family
- add stronger tracing and diagnostics for why a non-greedy candidate wins
- move beam search from local continuation scoring toward a fuller beam over
  partial-tree states
- define a stronger partial-tree score from:
  - current leaf objective
  - a heuristic estimate of future value
- deduplicate equivalent partial trees
- add beam budget controls:
  - maximum expansions
  - maximum memory
  - early stopping

### 15.2 Post-build non-greedy refinement

- build the tree greedily first
- revisit internal nodes one at a time while keeping topology fixed
- re-optimize split feature, threshold, and downstream routing
- recompute leaf weights after each accepted change
- stop after a fixed number of passes or when no more improvement is found

### 15.3 Optimal-search follow-up

- add a time-budgeted anytime mode that stops early if a wall-clock limit
  is hit and falls back to the best candidate found so far
- benchmark quality vs wall-clock tradeoff against lookahead and beam search
  at matched depth limits

### 15.4 Evaluation

- benchmark tree builders on:
  - training objective
  - validation objective
  - wall-clock time
  - tree size and realized depth
  - number of accepted refinements
- measure whether stronger non-greedy builders reduce the number of boosting
  rounds required to hit the same validation quality

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

### Soft trees

- add experimental soft-routing trees with probabilistic internal-node gates
- start with binary soft trees before considering multiway variants
- compare:
  - end-to-end differentiable training
  - structure-first training followed by soft re-optimization
- evaluate whether temperature or entropy regularization is needed to stop
  routing from collapsing too early
- decide whether inference should always evaluate the full weighted traversal or
  whether very small path probabilities can be pruned safely
- define export and runtime contracts explicitly, because soft routing is not
  compatible with the ordinary hard-branch semantics

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

The most likely causes are:

- histogram construction is still too expensive in the hottest paths
- feature access is not cache-friendly enough on wide binned tables
- too much per-node scratch rebuilding or temporary allocation remains
- feature subsampling does not cut off enough work early enough

The main improvements to target are:

- make histogram construction the dominant optimization focus
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

### Native categorical subset splits

Support grouped category tests like:

```text
x in {"red", "green"} -> left
x in {"blue"} -> right
```

Native subset split support for CART-style learners would require:

- a first-class categorical feature representation in the training table
- per-node category histograms instead of only numeric-bin histograms
- split metadata that can express "left category set" rather than only
  threshold comparisons
- runtime support for fast category-membership tests
- IR support for categorical split predicates and category vocabularies

That is likely the strongest long-term end state for medium-cardinality
categoricals.

### Leakage controls for target-style strategies

- add out-of-fold or CatBoost-style prefix statistics for target-informed
  categorical encoders
- ensure each row is encoded without leaking its own target contribution
- treat boosting as the highest-priority beneficiary, because stage-wise
  residual fitting is especially sensitive to weak spurious signal

### Canary-informed smoothing

- replace heuristic canary-informed categorical smoothing with stronger and
  more principled shrinkage rules
- focus especially on rare categories and high-cardinality columns in boosting
  settings

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

Boosting is a special case. In boosting, later trees are not trying to explain
the full target from scratch.
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

High-value coverage here includes:

- introspection parity across semantic and optimized forms
- broader parity coverage for newer surfaces such as categorical preprocessing
  once those contracts stabilize
- continued regression coverage whenever runtime-specialized features are added
