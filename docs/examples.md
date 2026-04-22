# Examples

These examples are meant to show the full workflow that ForestFire is built
around:

- train a model
- inspect the learned structure
- lower it for faster inference
- serialize it
- reload it
- score data in batch

The details vary by algorithm, but the overall shape is intentionally stable.

## Example 1: Random forest from training to deployment

This example uses a random forest classifier and walks through the most common
production-oriented lifecycle.

```python
import forestfire
import numpy as np

from forestfire import Model, Table, train

rng = np.random.default_rng(7)
n = 20_000

X = rng.normal(size=(n, 10))
y = (
    (X[:, 0] > 0.2)
    ^ (X[:, 1] > -0.1)
    ^ ((X[:, 2] + X[:, 3]) > 0.0)
).astype(float)

table = Table(X, y, canaries=0)

model = train(
    table,
    algorithm="rf",
    tree_type="cart",
    n_trees=1000,
    max_features="sqrt",
    min_samples_leaf=5,
    compute_oob=True,
)

print(model.oob_score)
print(model.tree_count)
print(model.tree_structure(tree_index=0))
print(model.tree_prediction_stats(tree_index=0))

tree_df = model.to_dataframe(tree_index=0)
print(tree_df)

optimized = model.optimize_inference()

# If only features 0 and 1 may be missing at inference time:
# optimized = model.optimize_inference(missing_features=[0, 1])

print(model.used_feature_indices)
print(optimized.used_feature_indices)

batch = X[:5_000]
proba = optimized.predict_proba(batch)
pred = optimized.predict(batch)

print(proba[:3])
print(pred[:3])

payload = model.serialize()
reloaded = Model.deserialize(payload)

reloaded_pred = reloaded.predict(batch)
print(np.allclose(pred, reloaded_pred))

compiled = optimized.serialize_compiled()
compiled_reloaded = forestfire.OptimizedModel.deserialize_compiled(compiled)
print(np.allclose(compiled_reloaded.predict(batch), pred))
```

What this example shows:

- `Table(...)` lets preprocessing and training be treated as separate steps
- training and introspection stay on the semantic model
- `optimize_inference()` creates a runtime-oriented scoring view without changing
  model meaning
- `missing_features=[...]` is available when only some columns need explicit
  missing checks in the optimized runtime
- used-feature metadata shows how much of the original feature space the model
  actually depends on
- serialization and reload preserve the same prediction semantics
- compiled optimized artifacts preserve the lowered runtime as well as the
  semantic model
- batch scoring is the intended normal usage mode once the model is trained

## Example 2: Gradient boosting with introspection and reload

This example uses gradient boosting to show the same lifecycle on a stage-wise
ensemble.

```python
import numpy as np

from forestfire import Model, train

rng = np.random.default_rng(19)
n = 30_000

X = rng.normal(size=(n, 12))
logit = (
    2.2 * (X[:, 0] > 0.4)
    - 1.8 * (X[:, 1] > -0.2)
    + 1.5 * ((X[:, 2] > 0.0) & (X[:, 3] < 0.5))
    + 0.9 * (X[:, 4] * X[:, 5] > 0.0)
    + 0.7 * X[:, 6]
)
p = 1.0 / (1.0 + np.exp(-logit))
y = rng.binomial(1, p).astype(float)

model = train(
    X,
    y,
    algorithm="gbm",
    tree_type="cart",
    learning_rate=0.05,
    n_trees=1000,
    top_gradient_fraction=0.2,
    other_gradient_fraction=0.1,
    canaries=2,
    filter=0.95,
)

print(model.tree_count)
print(model.tree_structure(tree_index=0))
print(model.tree_leaf(leaf_index=0, tree_index=0))

optimized = model.optimize_inference()

print(model.used_feature_count)
print(optimized.used_feature_count)

batch = X[:10_000]
base_pred = model.predict_proba(batch)
fast_pred = optimized.predict_proba(batch)

print(np.allclose(base_pred, fast_pred))

ir_json = model.to_ir_json()
print(ir_json[:200])

payload = model.serialize()
reloaded = Model.deserialize(payload)
print(np.allclose(reloaded.predict_proba(batch), base_pred))
```

What this example shows:

- boosting still fits into the same top-level lifecycle as trees and forests
- `tree_count` reflects the realized number of stages, which may be smaller than
  the requested `n_trees` because of canary-based stopping
- `filter=0.95` keeps canary competition active while allowing the best real
  root split to survive if it still lands inside the top 5% of ranked
  candidates
- introspection still works tree-by-tree on a boosted ensemble
- optimized inference is expected to preserve the probability path, not just the
  hard labels
- used-feature metadata is often especially informative for ensembles, because
  the runtime may only need a fraction of the original feature columns
- JSON IR export and binary serialization both operate on the same semantic
  model

## Example 3: Inspecting the canary `filter` policy directly

This example focuses only on the canary acceptance rule. The important idea is
that splits are still scored and sorted in the usual way first. `filter` only
controls how far down that ranked list ForestFire is allowed to look for the
first real feature.

```python
import numpy as np

from forestfire import train

rng = np.random.default_rng(23)
n = 8_000

X = rng.normal(size=(n, 6))
y = (
    1.8 * X[:, 0]
    - 0.9 * X[:, 1]
    + 0.4 * X[:, 2]
    + rng.normal(scale=0.8, size=n)
    > 0.0
).astype(float)

strict = train(
    X,
    y,
    task="classification",
    tree_type="cart",
    canaries=2,
)

top_3 = train(
    X,
    y,
    task="classification",
    tree_type="cart",
    canaries=2,
    filter=3,
)

top_5_percent = train(
    X,
    y,
    task="classification",
    tree_type="cart",
    canaries=2,
    filter=0.95,
)

print(strict.tree_structure())
print(top_3.tree_structure())
print(top_5_percent.tree_structure())
```

How to read this:

- the default model uses the strict policy, equivalent to `filter=1`
- `filter=3` allows the best real feature to be chosen as long as it is still
  within the top 3 ranked candidates
- `filter=0.95` allows the best real feature to be chosen as long as it is
  still within the top `ceil(5% * candidate_count)` ranked candidates

What this example shows:

- `filter` does not change how split quality is computed
- canaries may still outrank the chosen real feature
- the selected split is always the highest-ranked real feature inside the
  allowed window
- if the allowed window contains only canaries, the usual canary stop still
  happens

## Example 4: Oblique splits on a rotated boundary

This example shows oblique splits on a dataset whose decision boundary is
diagonal in feature space. It compares axis-aligned and oblique trees and then
walks through the same serialize-and-reload lifecycle.

```python
import numpy as np

from forestfire import Model, train

rng = np.random.default_rng(7)
n = 15_000

# The true rule is a rotated boundary: x_0 + x_1 > 0.5
# An axis-aligned tree must approximate this with many staircase splits.
# An oblique tree can express it in a single node.
X = rng.normal(size=(n, 8))
y = (X[:, 0] + X[:, 1] > 0.5).astype(float)

# Axis-aligned baseline
model_aa = train(
    X,
    y,
    algorithm="dt",
    tree_type="cart",
    task="classification",
    canaries=2,
)

# Oblique: learns a pairwise linear split w1 * x_i + w2 * x_j <= t
model_ob = train(
    X,
    y,
    algorithm="dt",
    tree_type="cart",
    task="classification",
    split_strategy="oblique",
    canaries=2,
)

print("axis-aligned tree:")
print(model_aa.tree_structure())

print("oblique tree:")
print(model_ob.tree_structure())

# Oblique splits are available for random forests and gradient boosting too
model_rf_ob = train(
    X,
    y,
    algorithm="rf",
    tree_type="cart",
    task="classification",
    split_strategy="oblique",
    n_trees=300,
    max_features="sqrt",
    min_samples_leaf=5,
)

model_gbm_ob = train(
    X,
    y,
    algorithm="gbm",
    tree_type="cart",
    task="classification",
    split_strategy="oblique",
    learning_rate=0.05,
    n_trees=500,
    canaries=2,
    filter=0.95,
)

# All four models follow the same serialize-reload-score lifecycle
batch = X[:5_000]

for label, model in [
    ("axis-aligned DT", model_aa),
    ("oblique DT", model_ob),
    ("oblique RF", model_rf_ob),
    ("oblique GBM", model_gbm_ob),
]:
    optimized = model.optimize_inference()
    pred = optimized.predict_proba(batch)

    payload = model.serialize()
    reloaded = Model.deserialize(payload)
    reloaded_pred = reloaded.predict_proba(batch)

    print(f"{label}: parity={np.allclose(pred, reloaded_pred)}, used_features={model.used_feature_count}")

# Oblique node structure is visible in introspection
print(model_ob.tree_node(node_id=0, tree_index=0))

# The IR records the participating feature indices, weights, threshold,
# and per-feature missing-direction metadata for each oblique node
ir = model_ob.to_ir_json()
print(ir[:300])
```

What this example shows:

- `split_strategy="oblique"` is a drop-in swap that does not change the rest
  of the API
- axis-aligned and oblique candidates compete in the same pool at every node,
  so the oblique model only introduces oblique nodes where they actually win
- the same optimized inference, serialization, and reload workflow applies to
  oblique models without any special handling
- oblique splits are visible in `tree_node(...)` and the JSON IR as
  `ObliqueLinearCombination` entries with two feature indices, two weights, and
  a threshold in the projected 1D space
- `used_feature_count` may differ between axis-aligned and oblique models
  because oblique nodes can cover two features in one split, and the winning
  competition may leave different features unused

## Why these examples matter

ForestFire is not only a trainer and not only an inference runtime.

Its architecture is built around one semantic model that can serve several
purposes:

- explain what was trained
- be serialized and moved
- be lowered into a faster runtime
- be reloaded and scored later

That is why the examples above deliberately go beyond `train(...)` and
`predict(...)`. The interesting part of the library is the continuity between
training, inspection, optimization, and deployment.
