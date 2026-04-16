# regression surface summary | gradient_boosting

This benchmark uses a two-dimensional non-linear synthetic regression task to compare fitted prediction surfaces.

Prediction plot: `regression_surface_gradient_boosting.png`

| backend | train | predict | rmse | r2 |
| --- | ---: | ---: | ---: | ---: |
| `forestfire_cart` | `0.000191s` | `0.000021s` | `0.4771` | `-0.0175` |
| `forestfire_randomized` | `0.000184s` | `0.000016s` | `0.4760` | `-0.0127` |
| `forestfire_oblivious` | `0.000344s` | `0.000014s` | `0.4800` | `-0.0297` |
| `sklearn` | `1.094246s` | `0.014174s` | `0.4533` | `0.0817` |
| `lightgbm` | `0.009519s` | `0.000964s` | `0.3167` | `0.5517` |
| `xgboost` | `0.018695s` | `0.000844s` | `0.4149` | `0.2306` |
| `catboost` | `0.156612s` | `0.000959s` | `0.3503` | `0.4516` |
