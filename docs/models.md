# Models And Introspection

## Prediction

- `predict(...)` returns hard predictions
- `predict_proba(...)` returns class probabilities for classification models

## Optimized inference

`optimize_inference(...)` lowers a trained model into a runtime-oriented representation for faster prediction.

It preserves model semantics while changing the execution layout.

## Serialization

Available export paths:

- JSON model serialization
- JSON IR export
- compiled optimized runtime serialization

## Tree introspection

All tree-backed models expose:

- tree structure summaries
- node/level/leaf inspection
- prediction-value statistics
- dataframe export

Typical use cases:

- understanding realized tree size and depth
- inspecting learned splits and leaf payloads
- comparing optimized and non-optimized views
- inspecting one tree at a time inside forests and boosted ensembles
