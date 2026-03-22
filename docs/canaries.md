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

- if a real feature wins, the tree can continue
- if the best winner is a canary, that node should stop

This matters because standalone trees are the most exposed to “keep splitting because you can” behavior.

In that setting, canaries let ForestFire be less dependent on post-hoc pruning.

## How canaries affect different tree families

The effect is not identical across tree families because the growth structure differs.

### ID3 and C4.5

These are multiway classifiers.

Here, canaries matter as a branch-expansion gate:

- if the best split at a node is effectively noise, growth stops there

This is useful because multiway branching can create a large structural jump from a single split.

### CART

For standard binary trees, the canary acts as the most direct local stop test.

This is the easiest setting to reason about:

- if the best binary split does not beat noise, stop

### Randomized trees

Randomized trees intentionally add stochasticity to split search.

That makes canaries especially useful, not less useful:

- randomness can expose different candidate splits
- canaries help decide whether those candidates are still meaningfully better than noise

### Oblivious trees

Oblivious trees grow by level, not by arbitrary node.

That changes the interpretation:

- a canary win does not just say “one node should stop”
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
2. the root split of the next stage would be a canary

That second rule is important.

It means:

- the stage is not merely weakened
- it is discarded entirely
- boosting stops at that point

This is a strong statement about the residual problem:

- if even the root of the next stage is best explained by noise, continuing to add trees is not justified

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
