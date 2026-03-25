# Training

## Algorithms

- `dt`: one tree
- `rf`: random forest
- `gbm`: gradient boosting

These are separate algorithms, not just cosmetic wrappers around the same tree builder:

- `dt` optimizes a single tree directly and is useful when interpretability matters most
- `rf` reduces variance by averaging many independently sampled trees
- `gbm` fits trees stage-by-stage to residual structure and is meant to trade interpretability for stronger predictive performance

## Support matrix

- `algorithm="dt"`
  - regression: `cart`, `randomized`, `oblivious`
  - classification: `id3`, `c45`, `cart`, `randomized`, `oblivious`
- `algorithm="rf"`
  - regression: `cart`, `randomized`, `oblivious`
  - classification: `id3`, `c45`, `cart`, `randomized`, `oblivious`
- `algorithm="gbm"`
  - regression: `cart`, `randomized`, `oblivious`
  - classification: `cart`, `randomized`, `oblivious` with binary targets only

The support matrix is narrower for boosting because boosting needs more than a split criterion and a leaf value. It needs consistent gradient/Hessian handling, additive stage updates, and stable leaf semantics under repeated residual fitting. That is why the current boosting support is focused on the binary-split regression-style tree families that map cleanly onto second-order stage training.

## Tree types

Classification:

- `id3`
- `c45`
- `cart`
- `randomized`
- `oblivious`

Regression:

- `cart`
- `randomized`
- `oblivious`

Why multiple tree types exist:

- `id3` and `c45` keep the classic multiway classifier style and are useful when you want a direct, explicit branch-per-bin structure
- `cart` is the general-purpose binary-tree baseline and the most natural fit for forests and boosting
- `randomized` introduces stochasticity into split search, which is useful both as a standalone learner and as the basis for extra-trees-style ensembles
- `oblivious` enforces one split per depth across the tree, which makes the final model more regular and easier to lower into compact runtime layouts

## Task detection

With `task="auto"`:

- integer, boolean, and string targets become classification
- float targets become regression

This keeps the common path ergonomic without hiding the important distinction that task choice changes:

- the split objective
- the leaf payload
- the valid tree families
- the prediction API surface, especially `predict_proba(...)`

## Stopping and control parameters

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `seed`

These parameters materially affect learning across the supported tree families, including forest constituent trees.

The rationale behind these controls is straightforward:

- `max_depth` limits structural complexity directly and is the cleanest way to cap traversal cost
- `min_samples_split` prevents very small unstable internal nodes
- `min_samples_leaf` prevents highly confident leaves built from tiny support
- `max_features` controls correlation and search width, which matters much more in ensembles than in a single tree
- `seed` is there because the library deliberately uses randomness in several places: bootstrapping, feature subsampling, randomized splits, and boosting row sampling

The intended semantics of `seed` are:

- fixed seed -> repeatable training for the same dataset and configuration
- different seed -> legitimately different stochastic choices

That randomness is not left ad hoc inside each learner. ForestFire now uses a
shared seed-mixing path for:

- forest tree seeds
- boosting stage seeds
- node-local feature-subset sampling
- randomized threshold selection

One practical detail matters here: node-local randomization is derived from the
rows owned by the node, but not from the incidental order of the mutable
row-index buffer. That makes the stochastic behavior more stable under internal
partitioning details while remaining sensitive to the actual training data seen
at each node.

## Canaries

ForestFire uses canary features to stop growth during training.

For the full rationale and algorithm-specific behavior, see [Canary Strategy](canaries.md).

Behavior:

- standard trees stop at the current node
- oblivious trees stop further depth growth
- gradient boosting stops adding stages if the next stage’s root split would be a canary
- random forests ignore canaries during tree training

Why canaries exist:

- standard tree learners can always continue to fit noise if the stopping rule only looks at local impurity decrease
- a canary feature is a shuffled copy of real signal space, so if the learner prefers it, the current split quality is no better than structured noise
- this turns stopping into a training-time competition between real and noise-like features instead of a later pruning pass

Why random forests ignore them:

- forests already reduce overfitting primarily through bootstrapping and feature subsampling
- reintroducing canaries inside each tree would often stop growth for the wrong reason, because each constituent tree already sees a deliberately restricted view of the data
- in boosting, the opposite tradeoff is useful: canaries remain active because stage-wise residual fitting is much more willing to chase noise late in training

## Random forests

- bootstrap sampling per tree
- feature subsampling per node
- optional OOB score computation

The random-forest implementation is intentionally close to the classical idea:

- each tree trains on a bootstrap sample
- each node considers only a sampled feature subset
- predictions are aggregated by averaging leaf outputs or probabilities

The important design choice is that preprocessing is still shared. Trees do not each rebucket the raw data. They all operate over the same preprocessed representation, which keeps training semantics aligned and avoids duplicated preprocessing work.

## Gradient boosting

- second-order trees trained from gradients and Hessians
- `learning_rate`
- optional stage bootstrapping
- LightGBM-style gradient-focused row sampling via:
  - `top_gradient_fraction`
  - `other_gradient_fraction`

The boosting implementation is LightGBM-like in spirit, but not a clone.

It keeps the ideas that matter most:

- second-order tree fitting from gradients and Hessians
- shrinkage via `learning_rate`
- optional row sampling
- focusing stage fitting on high-gradient rows

But it keeps a ForestFire-specific stopping rule:

- canaries still participate in split selection
- if the root split of the next stage would be a canary, that stage is discarded and boosting stops

That choice reflects the project’s general design preference: use explicit training-time noise competition as a stopping signal instead of bolting on a separate pruning or early-stopping layer later.

## Training optimizations

ForestFire training is optimized around a compact binned core and shared row-index buffers rather than row copies.

- numeric features are pre-binned into compact integer ranks, capped at `128` bins
- long-running training and prediction release the Python GIL before entering the Rust hot path
- CART and randomized trees use histogram-based numeric split search
- standard binary trees partition row indices in place
- ID3 and C4.5 use the same in-place row-buffer approach for multiway branches
- oblivious trees train over shared row buffers plus per-leaf ranges
- CART/randomized classification and mean-regression builders reuse parent histograms and derive sibling histograms by subtraction
- oblivious split scoring reuses cached per-leaf counts or `sum`/`sum_sq`
- random forests parallelize across trees while limiting intra-tree parallelism
- binary sparse inputs stay sparse through training and inference
- classifier, regressor, and second-order tree builders now share the same core
  histogram/partitioning/randomization helpers instead of carrying separate
  implementations of the same mechanics

Why these optimizations matter:

- binning turns repeated threshold search into a bounded discrete problem, which makes both scoring and runtime lowering much cheaper
- shared row-index buffers avoid per-node row-vector allocation, which is one of the main hidden costs in naive tree builders
- histogram subtraction matters because sibling statistics are not independent work; once one child is known, the other is often derivable from the parent
- shared tree internals matter because performance-sensitive fixes to histogram
  handling, partitioning, and stochastic split selection now land once and
  propagate across first-order and second-order learners together
- keeping sparse binary inputs sparse avoids paying a dense-memory penalty for data that is structurally sparse

Impact in practice:

- better cache behavior during split search
- less allocator pressure during recursive growth
- more predictable runtime layouts for optimized inference
- much better scaling in forests, where repeated tree training amplifies every avoidable allocation and recounting cost
