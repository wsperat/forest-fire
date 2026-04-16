# regression surface summary | gradient_boosting

This benchmark uses a two-dimensional non-linear synthetic regression task to compare fitted prediction surfaces.

Prediction plot: `regression_surface_gradient_boosting.png`

| backend | train | predict | rmse | r2 |
| --- | ---: | ---: | ---: | ---: |
| `forestfire_cart` | `0.046804s` | `0.003715s` | `0.0970` | `0.9652` |
| `forestfire_randomized` | `0.007951s` | `0.000564s` | `0.4225` | `0.3394` |
| `forestfire_oblivious` | `0.032689s` | `0.000879s` | `0.3353` | `0.5839` |
| `sklearn` | `9.733477s` | `0.013029s` | `0.0951` | `0.9666` |
| `lightgbm` | `0.175304s` | `0.029748s` | `0.0936` | `0.9675` |
| `xgboost` | `0.309910s` | `0.017886s` | `0.0933` | `0.9678` |
| `catboost` | `0.983449s` | `0.003383s` | `0.0878` | `0.9715` |
