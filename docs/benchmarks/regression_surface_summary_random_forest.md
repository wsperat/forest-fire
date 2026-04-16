# regression surface summary | random_forest

This benchmark uses a two-dimensional non-linear synthetic regression task to compare fitted prediction surfaces.

Prediction plot: `regression_surface_random_forest.png`

| backend | train | predict | rmse | r2 |
| --- | ---: | ---: | ---: | ---: |
| `forestfire_cart` | `3.792139s` | `0.338172s` | `0.0982` | `0.9636` |
| `forestfire_randomized` | `3.588657s` | `0.314749s` | `0.0917` | `0.9682` |
| `sklearn` | `2.607798s` | `0.252547s` | `0.0901` | `0.9693` |
| `lightgbm` | `15.563566s` | `0.814370s` | `0.0981` | `0.9636` |
| `xgboost` | `1.988891s` | `0.135612s` | `0.3695` | `0.4839` |
