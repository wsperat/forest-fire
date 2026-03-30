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
import polars as pl

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
- introspection still works tree-by-tree on a boosted ensemble
- optimized inference is expected to preserve the probability path, not just the
  hard labels
- used-feature metadata is often especially informative for ensembles, because
  the runtime may only need a fraction of the original feature columns
- JSON IR export and binary serialization both operate on the same semantic
  model

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
