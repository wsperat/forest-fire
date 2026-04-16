# regression curve summary | gradient_boosting

This benchmark uses a one-dimensional non-linear synthetic regression task to compare fitted prediction curves.

Prediction plot: `regression_curve_gradient_boosting.png`

| backend | train | predict | rmse | r2 |
| --- | ---: | ---: | ---: | ---: |
| `forestfire_cart` | `0.022256s` | `0.002670s` | `0.0906` | `0.9817` |
| `forestfire_randomized` | `0.003323s` | `0.000444s` | `0.4650` | `0.5169` |
| `forestfire_oblivious` | `0.049213s` | `0.001389s` | `0.1519` | `0.9485` |
| `sklearn` | `2.720321s` | `0.014118s` | `0.0827` | `0.9847` |
| `lightgbm` | `0.046465s` | `0.043862s` | `0.0828` | `0.9847` |
| `xgboost` | `0.042520s` | `0.012210s` | `0.0837` | `0.9844` |
| `catboost` | `0.297929s` | `0.002461s` | `0.0833` | `0.9845` |
