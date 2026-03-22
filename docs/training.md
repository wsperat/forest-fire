# Training

## Algorithms

- `dt`: one tree
- `rf`: random forest
- `gbm`: gradient boosting

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

## Task detection

With `task="auto"`:

- integer, boolean, and string targets become classification
- float targets become regression

## Stopping and control parameters

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `seed`

## Canaries

ForestFire uses canary features to stop growth during training.

Behavior:

- standard trees stop at the current node
- oblivious trees stop further depth growth
- gradient boosting stops adding stages if the next stage’s root split would be a canary
- random forests ignore canaries during tree training

## Random forests

- bootstrap sampling per tree
- feature subsampling per node
- optional OOB score computation

## Gradient boosting

- second-order trees trained from gradients and Hessians
- `learning_rate`
- optional stage bootstrapping
- LightGBM-style gradient-focused row sampling via:
  - `top_gradient_fraction`
  - `other_gradient_fraction`
