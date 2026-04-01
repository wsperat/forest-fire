# make_moons summary | random_forest

This benchmark uses sklearn.make_moons to compare probability surfaces on a two-dimensional non-linear classification task.

Probability plot: `moons_probabilities_random_forest.png`

| backend | train | predict | accuracy | log loss |
| --- | ---: | ---: | ---: | ---: |
| `forestfire_cart` | `0.221766s` | `0.021367s` | `0.9185` | `0.2953` |
| `forestfire_oblivious` | `0.616203s` | `0.013088s` | `0.9219` | `0.2936` |
| `sklearn` | `0.457130s` | `0.017896s` | `0.9458` | `0.1529` |
| `lightgbm` | `0.122228s` | `0.021035s` | `0.9453` | `0.2511` |
| `xgboost` | `0.135362s` | `0.014131s` | `0.9463` | `0.4915` |
