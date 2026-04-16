# make_moons summary | random_forest

This benchmark uses sklearn.make_moons to compare probability surfaces on a two-dimensional non-linear classification task.

Probability plot: `moons_probabilities_random_forest.png`

| backend | train | predict | accuracy | log loss |
| --- | ---: | ---: | ---: | ---: |
| `forestfire_cart` | `2.735222s` | `0.036121s` | `0.9233` | `0.4151` |
| `forestfire_randomized` | `3.781470s` | `0.041747s` | `0.9395` | `0.1795` |
| `sklearn` | `0.672062s` | `0.022418s` | `0.9419` | `0.2814` |
| `lightgbm` | `0.402532s` | `0.039737s` | `0.9390` | `0.2350` |
| `xgboost` | `0.147506s` | `0.012896s` | `0.9458` | `0.4905` |
