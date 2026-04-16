# make_moons summary | random_forest

This benchmark uses sklearn.make_moons to compare probability surfaces on a two-dimensional non-linear classification task.

Probability plot: `moons_probabilities_random_forest.png`

| backend | train | predict | accuracy | log loss |
| --- | ---: | ---: | ---: | ---: |
| `forestfire_cart` | `2.819376s` | `0.039456s` | `0.9233` | `0.4151` |
| `forestfire_randomized` | `4.011775s` | `0.045686s` | `0.9395` | `0.1795` |
| `sklearn` | `0.534519s` | `0.024532s` | `0.9419` | `0.2814` |
| `lightgbm` | `0.441031s` | `0.044241s` | `0.9390` | `0.2350` |
| `xgboost` | `0.160633s` | `0.014220s` | `0.9458` | `0.4905` |
