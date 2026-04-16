# regression surface summary | random_forest

This benchmark uses a two-dimensional non-linear synthetic regression task to compare fitted prediction surfaces.

Prediction plot: `regression_surface_random_forest.png`

| backend | train | predict | rmse | r2 |
| --- | ---: | ---: | ---: | ---: |
| `forestfire_cart` | `1.677750s` | `0.061290s` | `0.0969` | `0.9652` |
| `forestfire_randomized` | `1.643910s` | `0.055705s` | `0.0923` | `0.9684` |
| `forestfire_oblivious` | `0.435815s` | `0.007086s` | `0.1626` | `0.9021` |
| `sklearn` | `1.037801s` | `0.052362s` | `0.0907` | `0.9696` |
| `lightgbm` | `5.725817s` | `0.104122s` | `0.0967` | `0.9654` |
| `xgboost` | `0.900750s` | `0.025081s` | `0.3766` | `0.4751` |
