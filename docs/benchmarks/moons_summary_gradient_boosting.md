# make_moons summary | gradient_boosting

This benchmark uses sklearn.make_moons to compare probability surfaces on a two-dimensional non-linear classification task.

Probability plot: `moons_probabilities_gradient_boosting.png`

| backend | train | predict | accuracy | log loss |
| --- | ---: | ---: | ---: | ---: |
| `forestfire_cart` | `0.032686s` | `0.002950s` | `0.9448` | `0.1546` |
| `forestfire_randomized` | `0.032386s` | `0.002534s` | `0.9458` | `0.1910` |
| `forestfire_oblivious` | `0.032831s` | `0.001015s` | `0.9458` | `0.1920` |
| `sklearn` | `0.636058s` | `0.009227s` | `0.9429` | `0.1673` |
| `lightgbm` | `0.031769s` | `0.012736s` | `0.9429` | `0.1652` |
| `xgboost` | `0.032179s` | `0.010188s` | `0.9453` | `0.1585` |
| `catboost` | `0.335256s` | `0.002575s` | `0.9429` | `0.1519` |
