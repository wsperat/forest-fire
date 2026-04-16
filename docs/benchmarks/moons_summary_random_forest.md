# make_moons summary | random_forest

This benchmark uses sklearn.make_moons to compare probability surfaces on a two-dimensional non-linear classification task.

Probability plot: `moons_probabilities_random_forest.png`

| backend | train | predict | accuracy | log loss |
| --- | ---: | ---: | ---: | ---: |
| `forestfire_cart` | `0.356772s` | `0.068455s` | `0.8940` | `0.4947` |
| `forestfire_oblivious` | `0.290355s` | `0.056405s` | `0.8950` | `0.5027` |
| `sklearn` | `0.888049s` | `0.054626s` | `0.9114` | `0.2368` |
| `lightgbm` | `0.113960s` | `0.055991s` | `0.9102` | `0.3273` |
| `xgboost` | `0.090129s` | `0.056261s` | `0.9112` | `0.5173` |
