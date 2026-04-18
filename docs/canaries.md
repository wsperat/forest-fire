# Canary Strategy

ForestFire uses canary features as a training-time noise baseline during tree growth.

This is one of the project’s strongest design opinions, and it affects how you should think about stopping, regularization, and ensemble behavior.

## What a canary is

A canary feature is a shuffled copy of an already-preprocessed real feature.

That means:

- it looks statistically plausible
- it has the same value domain as a real feature
- it carries no real relationship to the target

The point is not to create obviously fake noise. The point is to create a realistic competitor that answers a useful question:

> is the best available split better than structured noise?

If the answer is no, the learner should usually stop.

## Why ForestFire uses canaries

Classic tree growth has a familiar problem:

- if you keep searching for local impurity improvement, you can usually keep finding small “improvements” in noise
- those improvements may be mathematically real on the training sample while still being statistically meaningless

A canary turns that into a direct competition.

Instead of asking only:

- “does this split improve the objective?”

ForestFire also effectively asks:

- “does this split beat a shuffled feature drawn from the same feature space?”

That changes the role of stopping.

Stopping is no longer just:

- max depth
- minimum samples
- minimum gain
- post-hoc pruning

It becomes part of split selection itself.

## What canaries replace

Canaries partially replace several traditional “don’t let the tree grow too far” mechanisms.

In practice they take over part of the job usually handled by:

- pruning
- overly aggressive depth caps
- large minimum-leaf settings used purely as a noise guard
- minimum-gain heuristics used only to prevent tiny accidental improvements

Why this is useful:

- the stopping signal is data-dependent
- the threshold is contextual rather than globally fixed
- the learner compares real features to realistic noise, not to an arbitrary constant

That makes the stop decision more adaptive than saying:

- “never go deeper than `d`”
- “never split below `n` rows”
- “never accept gain below `g`”

Those controls are still useful, but they are not trying to answer exactly the same question.

## What canaries do not replace

Canaries do **not** make the standard structural hyperparameters obsolete.

They do not replace:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `seed`

Why not:

### `max_depth`

`max_depth` caps complexity directly.

Even if every split is better than a canary, you may still want:

- bounded latency
- bounded model size
- bounded explanation depth

Canaries are about “is this split better than noise?”

`max_depth` is about “how large am I willing to let this object become?”

Those are different questions.

### `min_samples_split`

This prevents the learner from even considering splits in nodes that are too small to support stable estimates.

Canaries help judge signal quality, but they do not repair the fact that tiny nodes are high-variance objects.

### `min_samples_leaf`

This constrains how concentrated confidence can become.

Even if a split beats a canary, a leaf with only a handful of rows may still be operationally undesirable for:

- calibration
- interpretability
- robustness

### `max_features`

This changes the search space itself.

Canaries decide whether growth should continue. `max_features` decides how wide the competition set is at each node or level.

That matters especially in ensembles.

### `seed`

Canaries are shuffled copies. Randomized trees, forests, and boosting also use randomness elsewhere.

If you care about reproducibility, you still need the seed.

## How canaries affect standalone trees

For ordinary `dt` training, canaries act as a local growth guard.

Effectively:

- if an acceptable real feature survives the canary policy, the tree can continue
- if the allowed competition window contains only canaries, that node should stop

This matters because standalone trees are the most exposed to “keep splitting because you can” behavior.

In that setting, canaries let ForestFire be less dependent on post-hoc pruning.

## Categorical variables

Canaries now also play a second role for categorical handling.

For ordinary numeric and binary training, canaries act mainly as a split
acceptance baseline:

- is this candidate split better than structured noise?

For categorical encodings, there is an additional failure mode:

- a target-informed encoding can manufacture convincing-looking structure from
  sparse or noisy categories even before the tree starts choosing splits

That matters most for:

- `target`
- `fisher`

and much less cleanly for:

- `dummy`

### Why categorical encodings need extra caution

`target` and `fisher` both use the training target while building the feature
representation itself.

That means a noisy categorical column can still look strong:

- `target` may give rare categories extreme encoded values
- `fisher` may produce a clean-looking category ordering that is mostly luck

If the library only asked whether the resulting transformed feature can win a
split later, it would be evaluating the encoding after some of the overfit risk
has already been baked in.

### Current behavior

ForestFire now uses canary-informed shrinkage for:

- `target`
- `fisher`

The basic idea is:

1. measure how much target-structured separation the real categorical feature
   appears to have under the current smoothing level
2. build a categorical canary by shuffling that feature while keeping the same
   category vocabulary and marginal counts
3. measure the same target-structured separation on the shuffled feature
4. increase smoothing when the canary remains too competitive with the real
   feature

So the canary is not just competing with the feature at split time. It is also
used earlier as a robustness check on the encoding strength itself.

### What gets adjusted

For `target`:

- the effective smoothing can be increased on a per-feature basis

For `fisher`:

- the same canary-informed smoothing adjustment is applied before computing the
  target-derived category ordering

That means weak or fragile category signal gets pulled harder toward the global
prior before the downstream tree learner ever sees the transformed feature.

The base meaning of `target_smoothing` belongs to the categorical-feature
documentation. The important canary-specific point is narrower:

- canaries may increase the effective smoothing above the user-provided base
  level when shuffled-category signal looks too competitive

### Why `dummy` is excluded

`dummy` is intentionally left out of this mechanism for now.

The reason is not that canaries are irrelevant there. The reason is that the
comparison unit is awkward:

- one source categorical feature expands into many derived indicator columns
- a naive per-indicator canary comparison would not be statistically matched to
  the source feature as a whole

So `dummy` would need a grouped or source-feature-level canary treatment rather
than the per-feature shrinkage rule used for `target` and `fisher`.

### What this does and does not mean

This is a robustness mechanism, not a guarantee that leakage-like effects are
fully solved.

It helps by making the encoding more conservative when shuffled categories can
still produce suspiciously strong apparent signal. But it is still a heuristic
shrinkage rule, not a full cross-fitting or ordered-statistics categorical
scheme.

In practical terms, the current design means:

- categorical canaries influence representation strength for `target` and
  `fisher`
- ordinary canary competition still governs whether splits are allowed to win
  later in tree growth
- `dummy` still uses the regular split-time canary mechanism only

## The windowed canary policy

The important point is that ForestFire now separates two ideas that used to be fused together:

- how candidates are scored and ranked
- how strict the canary stopping rule is

Scoring still happens exactly the same way:

- evaluate candidate splits
- assign each candidate its usual objective score
- sort candidates from best to worst

What `filter` changes is only the acceptance rule that runs after ranking.

### Default behavior: `filter=None` or `filter=1`

The default policy is intentionally strict.

It means:

- only the single top-ranked candidate is eligible
- if that candidate is a real feature, training continues with it
- if that candidate is a canary, the node or stage stops

This is the original “best split must beat noise directly” rule.

### Integer windows: `filter=n`

If you pass an integer `n`, ForestFire allows a slightly softer rule:

- rank all scored candidates as usual
- inspect only the top `n`
- choose the highest-ranked real feature inside that window

This means canaries are still allowed to rank ahead of the chosen real feature.

Example:

- `filter=3` means a canary may occupy rank 1
- another canary may occupy rank 2
- the best real feature at rank 3 is still allowed to split the node

If every candidate in the top 3 is a canary, the node still stops.

So `filter=n` is not “ignore canaries.”

It is:

- “keep canaries as a guardrail”
- “but allow the best real split to survive if it is still near the very top”

### Fractional windows: `filter=alpha`

If you pass a float `alpha` in `[0, 1)`, ForestFire interprets it as a top-window fraction of `1 - alpha`.

Concretely:

- let `k` be the number of scored candidates at the current decision point
- compute `ceil((1 - alpha) * k)`
- search only that top-ranked window for the first real feature

Example:

- `filter=0.95` means “accept the best real split inside the top 5% of ranked candidates”

That is useful when the number of scored candidates changes from node to node or from algorithm to algorithm and you want the canary rule to scale with the actual competition set rather than with a fixed absolute count.

### What the window is counting

The `filter` window is computed over scored split candidates, not over the original raw columns in isolation.

That distinction matters because the candidate pool depends on context:

- `max_features` may restrict which features are even considered
- some features may be unavailable or unsplittable at a given node
- oblivious trees score one shared level-split rather than independent per-node splits
- boosting applies the same logic only at the root of the next stage

So the meaning of “top 5%” is:

- top 5% of the candidates that were actually scored at that decision point

not:

- top 5% of all features in the original dataset regardless of feasibility

### What happens when the window contains only canaries

The answer is still “stop.”

That is the core invariant ForestFire keeps:

- canaries remain part of the acceptance test
- the filter only controls how much room a near-top real feature is given to survive
- if no real feature survives inside that allowed window, growth is rejected

This keeps the canary mechanism meaningful while allowing a less all-or-nothing policy than “rank 1 must already be real.”

## How canaries affect different tree families

The effect is not identical across tree families because the growth structure differs.

### ID3 and C4.5

These are multiway classifiers.

Here, canaries matter as a branch-expansion gate:

- if no real split survives inside the allowed top window, growth stops there

This is useful because multiway branching can create a large structural jump from a single split.

### CART

For standard binary trees, the canary acts as the most direct local stop test.

This is the easiest setting to reason about:

- if the allowed top-ranked window contains no real binary split, stop

### Randomized trees

Randomized trees intentionally add stochasticity to split search.

That makes canaries especially useful, not less useful:

- randomness can expose different candidate splits
- canaries help decide whether those candidates are still meaningfully better than noise

The `filter` window is especially useful here because randomized search can move near-tied candidates around more than fully deterministic search does. A slightly wider window lets you keep the same “real split must still be near the top” principle without requiring the real split to win rank 1 every time.

### Oblivious trees

Oblivious trees grow by level, not by arbitrary node.

That changes the interpretation:

- a canary-only window does not just say “one node should stop”
- it says “the next shared depth-level split is not justified”

So for oblivious trees, canaries naturally act as a depth-growth stop.

That is one reason they fit the oblivious setting well: the stopping signal is aligned with the level-wise growth pattern.

## How canaries interact with random forests

Random forests are the one place where ForestFire intentionally disables canaries during tree training.

That is not an omission. It is a design choice.

Why:

- forests already regularize heavily through bootstrapping
- they also regularize through feature subsampling
- each tree is intentionally weak, noisy, and diverse

If canaries were left active inside every forest tree, they would often stop growth for the wrong reason:

- not because the data has no usable structure
- but because each tree is already working with a restricted, noisy view of the problem

That would reduce useful ensemble diversity.

So in ForestFire:

- standalone trees use canaries
- random forests ignore them

This is why you should not interpret the absence of canaries in RF as inconsistent. It reflects a different regularization regime.

### What replaces canaries in random forests

In random forests, the main anti-overfitting forces are:

- bootstrap sampling
- `max_features`
- averaging across many trees
- ordinary structural limits like depth and minimum samples

Those are stronger and more appropriate regularizers for bagged ensembles than canary competition inside each constituent tree.

## How canaries interact with gradient boosting

Gradient boosting is the opposite case.

ForestFire keeps canaries active there on purpose.

Why:

- boosting fits trees stage by stage to what the current ensemble still gets wrong
- later stages are especially willing to chase tiny residual structure
- that is exactly where “signal vs structured noise” becomes most important

So in boosting, canaries are not disabled. They become an early-stop signal for stage addition.

## The boosting-specific rule

In ForestFire’s gradient boosting implementation, training stops when:

1. `n_trees` stages have been trained, or
2. the allowed root-level `filter` window contains no real split and only canaries survive there

That second rule is important.

It means:

- the stage is not merely weakened
- it is discarded entirely
- boosting stops at that point

This is a strong statement about the residual problem:

- if even the allowed top-ranked root candidates do not contain a real split that survives canary competition, continuing to add trees is not justified

## Why the root matters in boosting

The root split is the cleanest stage-level test.

If the first split of a new tree cannot beat a canary, then the residual signal the stage is trying to capture is already below the noise baseline at the coarsest possible level.

That makes it a natural stopping rule for additive boosting.

It is also much cheaper and easier to reason about than trying to define a later, more entangled “the whole stage was too weak” criterion.

## How this differs from classic early stopping

Many boosting systems rely on:

- validation loss tracking
- patience
- explicit early stopping rounds

ForestFire’s canary logic is different.

It is:

- training-internal
- structure-aware
- based on noise competition, not just held-out loss motion

That does not make validation-based early stopping useless. It means the library’s first line of defense is embedded directly in how the next stage is justified.

## Practical implications by algorithm

### Single trees

Canaries make the tree less likely to keep growing just because local impurity still improves slightly.

### Random forests

Canaries are disabled because bagging and feature subsampling are already the intended anti-overfitting mechanism.

### Gradient boosting

Canaries remain active and become a stage-level stop signal because boosting is especially prone to late-stage noise fitting.

## How to think about hyperparameters when canaries exist

A useful mental model is:

- canaries answer “is there still signal here?”
- structural hyperparameters answer “how large, stable, and expensive am I willing to let the model become?”

That division is why both exist.

You should not treat canaries as a total replacement for ordinary controls. You should treat them as a better answer to one specific problem:

- distinguishing real split signal from noise-like split opportunity

## Why this is a design choice, not just a trick

The canary strategy reflects ForestFire’s broader design philosophy:

- make semantics explicit
- push regularization into training-time decision points
- prefer noise-aware growth rules over “grow first, clean up later”

That is why canaries are not an optional afterthought in the codebase.

They are part of how the library defines justified tree growth.
