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
- standard second-order CART/randomized trees now use an active-node frontier,
  so a whole depth is evaluated before any node at that depth partitions the
  shared row-index buffer
- second-order histogram construction now has a parallel-capable shared helper
  instead of forcing the GBM path through a purely sequential per-feature build

That changes the next concrete implementation step. The structural batching
boundary now exists, so the next work should focus on using it:

- parallelize histogram construction across batches of active nodes
- parallelize split scoring across features within those node batches
- keep row partitioning as the later mutation step after split selection

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

## Add categorical-feature support for CART, random forests, and boosting

Categorical support is one of the clearest remaining training and model-format
gaps.

ForestFire already has explicit numeric and binary preprocessing, and IR v1
still states that categorical encodings are not implemented. That means this is
not just a trainer tweak. It touches:

- training-time feature representation
- split-search semantics
- missing and unknown-category handling
- optimized runtime execution
- IR/export metadata
- Python and Rust API contracts

It is also important to separate this roadmap from the existing `id3` / `c45`
story. Those tree families already allow multiway branching over
"categorical-like" binned features. The harder unsolved problem is categorical
support for the CART-style binary-split learners that also underpin random
forests and gradient boosting.

There is no single universally correct categorical strategy. In practice, tree
systems use several different ones depending on:

- whether the feature is nominal or truly ordered
- how many distinct categories it has
- whether the learner is a single tree, a forest, or a stage-wise boosted model
- whether the implementation has native categorical split search

The main strategies worth supporting or documenting are:

### 1. Native categorical subset splits

This is the most direct semantic match for CART-style trees.

For a nominal feature with `k` categories, the natural binary split is not a
numeric threshold like `x <= t`. It is a set-membership test:

- go left if `x in S`
- go right if `x not in S`

where `S` is some subset of the observed categories.

That matters because nominal categories do not have a meaningful numeric order.
If a feature contains `{"red", "blue", "green"}`, the useful question is often
"should `red` and `green` be grouped together against `blue`?" rather than
"is the encoded integer less than `1.5`?"

For plain CART and random forests, this native subset-split view is the clean
ideal because it preserves the true hypothesis class:

- one node can isolate a useful grouping of categories directly
- there is no fake distance notion between category ids
- the resulting tree structure matches how users think about categorical rules

The difficulty is split-search cost. A `k`-level categorical feature has
`2^(k-1) - 1` distinct binary partitions if complements are treated as the same
split. That is manageable for very small `k`, but it becomes too expensive to
search exhaustively once cardinality grows.

That means a serious implementation needs more than just a categorical column
type:

- exact subset search for very low-cardinality features
- heuristics or restricted search once cardinality becomes moderate
- minimum-support rules so tiny categories do not generate unstable splits
- a clear policy for categories never seen during training

For gradient boosting, native categorical support usually becomes
histogram-based rather than exhaustive. The common pattern is:

1. aggregate per-category gradient/Hessian statistics
2. order categories by some target-like summary derived from those statistics
3. scan only contiguous partitions in that ordered list instead of all possible
   subsets

That is the core idea behind the efficient categorical handling used by systems
like LightGBM and XGBoost. The important point is that the learner still
produces a native categorical split semantically, even though the search is
implemented through an ordering trick that avoids the full combinatorial search.

For ForestFire specifically, native categorical subset splits would require:

- a first-class categorical feature representation in the training table
- per-node category histograms instead of only numeric-bin histograms
- split metadata that can express "left category set" rather than only
  threshold comparisons
- runtime support for fast category-membership tests
- IR support for categorical split predicates and category vocabularies

This is likely the strongest long-term end state for medium-cardinality
categoricals because it gives CART, random forests, and GBM a natural and
expressive treatment without inflating feature width.

### 2. One-hot encoding

One-hot encoding is the lowest-risk compatibility path when native categorical
split search does not exist yet.

A categorical feature with levels such as `{"red", "blue", "green"}` is
expanded into multiple binary indicators:

- `is_red`
- `is_blue`
- `is_green`

or sometimes `k - 1` indicators plus an implicit baseline level.

The main reason this is attractive for ForestFire is that the system already
has explicit binary-feature handling. That means one-hot support could reuse a
large amount of the existing training and runtime machinery:

- binary features already fit the current split model naturally
- histogram and partitioning logic for binary columns already exists
- IR support for booleans is already much closer to complete than IR support for
  categorical predicates

This makes one-hot encoding the easiest near-term path for low-cardinality
categoricals.

The costs are equally important:

- feature width grows linearly with the number of categories
- training cost grows because more columns now compete at each node
- `max_features` semantics become less intuitive because one original feature
  may explode into many derived binary columns
- a grouped category rule may now require several tree levels instead of one
  native split

That last point is easy to underestimate. Suppose the useful rule is
"`red` or `green` means left, `blue` means right." A native categorical split
can represent that in one node. A one-hot representation often needs multiple
binary decisions across different nodes to express the same logic.

This is why one-hot encoding is usually best viewed as a pragmatic baseline, not
the ideal long-term representation.

The implementation details also matter:

- low-cardinality features should probably be the only automatic one-hot
  candidates
- missing and unseen categories should not silently collapse into ordinary
  all-zero rows unless that behavior is documented as intentional
- the model contract should record the category vocabulary and expansion mapping
  if the encoding happens inside training rather than outside the library

One-hot is therefore attractive as a staged first step:

- it gives users a correct nominal treatment for small category sets
- it leverages the current binary infrastructure
- it avoids blocking all categorical support on a larger native-split rewrite

### 3. Ordinal or integer encoding

Integer-coded categories are useful, but only under very specific semantics.

There are two very different situations that are often mistakenly grouped
together:

- truly ordered categories, such as `{"small", "medium", "large"}`
- arbitrary category ids, such as `{"red", "blue", "green"}` encoded as
  `{0, 1, 2}`

For genuinely ordered categories, ordinal encoding is reasonable. A split like
"`size <= medium`" carries real meaning. In that case, a threshold-based CART
learner is using a true order already present in the data.

For arbitrary nominal categories, plain ordinal encoding is dangerous. A CART
split over integer ids imposes a fake geometry:

- category `2` is treated as "greater than" category `1`
- categories with adjacent ids are treated as more similar than distant ids
- threshold search is forced to cut the categories into contiguous numeric
  blocks, even though the useful grouping may be non-contiguous in id space

That means a naïve `OrdinalEncoder` plus ordinary CART thresholding is usually
the wrong thing for nominal features.

The subtle but important exception is native categorical implementations that
store categories as integers internally while still treating them as labels
semantically. LightGBM, XGBoost, and similar systems often expect category ids
to be integer-coded in memory, but that does not mean they are doing ordinary
numeric thresholding over those ids. The integer code is only a compact storage
format for a categorical learner.

That distinction is important for ForestFire:

- integer coding may be the right internal representation for category values
- it should not automatically imply threshold-based numeric split search
- user-visible APIs should only allow pure ordinal treatment when the feature is
  explicitly declared ordered

In practice, ordinal encoding should probably appear in the roadmap in only two
forms:

- as an explicit feature type for genuinely ordered categories
- as an internal storage detail for native categorical split search

It should not be the default fallback for arbitrary string or enum-like
features, because that would silently give incorrect semantics while appearing
to "support categoricals."

### 4. Target, mean, and CTR-style encodings

Target-based encodings are often the strongest preprocessing option for
high-cardinality categoricals when native subset-split search is unavailable or
too expensive.

The idea is to replace each category with a target-derived statistic:

- for regression, a smoothed category mean
- for binary classification, a smoothed positive-class rate or log-odds
- for multiclass tasks, per-class or otherwise structured target statistics

This turns a high-cardinality categorical feature into one or a few numeric
features that the existing CART machinery can consume directly.

Why this works well:

- the representation size no longer grows with cardinality the way one-hot does
- rare categories can borrow strength from the global target prior through
  smoothing
- the learner can often extract strong signal from one numeric statistic rather
  than many sparse indicators

Why it is risky:

- naïve target encoding leaks label information badly
- the same category may appear in both the statistic and the row being encoded
- leakage is especially harmful in boosting, where residual fitting is already
  sensitive to weak accidental signal

So a credible implementation cannot be "compute category means on the full
training set and call it done."

It needs one of the standard leakage controls:

- out-of-fold or cross-fitted target statistics
- ordered or prefix-style statistics of the kind popularized by CatBoost
- strong smoothing and minimum-frequency rules
- a global-prior fallback for unseen or extremely rare categories

The CatBoost lesson is especially important. Its categorical handling is not
just "target encoding plus boosting." The key idea is that the statistics are
computed in an order-aware way so each row is encoded only from earlier rows,
which sharply reduces target leakage compared with full-dataset means.

For ForestFire, target-style encodings would be attractive for very
high-cardinality features because they fit the current numeric histogram stack
much more naturally than full subset-split search. But they come with model
contract implications:

- the encoding recipe becomes part of preprocessing semantics
- exported IR must record the prior, smoothing, category statistics, and
  unknown-category fallback
- training/test parity depends on preserving exactly how those encodings were
  produced

This means target encodings are not just an optional preprocessing helper if
they live inside the library. They become part of the learned model semantics.

### Strategy-specific fit by model family

The strategies above are not equally attractive for every tree family.

For one standalone CART tree:

- native subset splits are the cleanest nominal treatment
- one-hot is acceptable for low-cardinality features
- target encoding can help on very high-cardinality data
- naïve ordinal encoding should be limited to truly ordered features

For random forests:

- the same categorical choices apply at the tree level
- but high-cardinality features need extra caution because they offer many more
  candidate splits and can be over-selected
- one-hot can also distort `max_features` behavior by letting one source feature
  dominate the sampled candidate pool after expansion

For gradient boosting:

- native histogram-based categorical search is usually the best medium-card
  solution
- target or CTR-style encodings are often the strongest high-card solution
- leakage control matters much more because stage-wise fitting will exploit even
  weak spurious signal
- plain one-hot can still work for low-cardinality features, but it is often
  less elegant and less efficient than native support

### A caution that applies regardless of strategy

Categorical support is not only about feature representation. It also changes
the statistical behavior of split search.

High-cardinality features are often favored simply because they create more
candidate partitions. That can lead to:

- overly optimistic impurity gains
- unstable deep-tree behavior on rare categories
- biased feature-importance conclusions

That means any serious categorical roadmap should include guardrails such as:

- minimum category frequency thresholds
- smoothing or shrinkage for target-style encodings
- possibly restricted subset search for medium/high-cardinality features
- benchmarks that specifically vary category cardinality and rare-level
  frequency

This is especially important for random forests, where split-frequency-based
importance can become misleading when a high-cardinality feature wins often for
purely combinatorial reasons.

### The most practical staged plan

The cleanest roadmap is probably not "pick one categorical strategy forever."
It is to support different strategies for different parts of the problem.

The most practical order is:

1. add low-cardinality one-hot support as a compatibility layer that reuses the
   current binary-feature pipeline
2. add explicit ordered-category support so truly ordinal features can use
   threshold splits honestly
3. add native categorical subset splits for CART and random forests, with exact
   search only for very low cardinalities and restricted search beyond that
4. add histogram-based native categorical split search for the second-order GBM
   path
5. add leakage-controlled target / CTR encodings for very high-cardinality
   features
6. extend IR, optimized runtime, and tests so categorical semantics survive
   export, reload, and inference parity checks

That staged plan matches the real shape of the problem:

- one-hot is the easiest short-term unlock
- native subset splits are the best semantic end state for many categorical
  features
- target-style encodings are still necessary once cardinality gets large enough
  that direct subset search becomes awkward or statistically brittle

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
