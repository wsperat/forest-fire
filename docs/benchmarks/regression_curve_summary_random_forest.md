# regression curve summary | random_forest

This benchmark uses a one-dimensional non-linear synthetic regression task to compare fitted prediction curves.

Prediction plot: `regression_curve_random_forest.png`

| backend | train | predict | rmse | r2 |
| --- | ---: | ---: | ---: | ---: |
| `forestfire_cart` | `0.196450s` | `0.024898s` | `0.0913` | `0.9814` |
| `forestfire_randomized` | `0.171979s` | `0.020059s` | `0.1060` | `0.9749` |
| `forestfire_oblivious` | `0.274387s` | `0.007073s` | `0.1429` | `0.9544` |
| `sklearn` | `0.422135s` | `0.022502s` | `0.0826` | `0.9847` |
| `lightgbm` | `0.165191s` | `0.029084s` | `0.0825` | `0.9848` |
| `xgboost` | `0.096498s` | `0.010430s` | `0.4740` | `0.4980` |
