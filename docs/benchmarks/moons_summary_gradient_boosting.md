# make_moons summary | gradient_boosting

This benchmark uses sklearn.make_moons to compare probability surfaces on a two-dimensional non-linear classification task.

Probability plot: `moons_probabilities_gradient_boosting.png`

| backend | train | predict | accuracy | log loss |
| --- | ---: | ---: | ---: | ---: |
| `forestfire_cart` | `0.033153s` | `0.003082s` | `0.9448` | `0.1546` |
| `forestfire_randomized` | `0.033208s` | `0.002609s` | `0.9458` | `0.1910` |
| `forestfire_oblivious` | `0.054168s` | `0.001101s` | `0.9463` | `0.1559` |
| `sklearn` | `1.085889s` | `0.013969s` | `0.9429` | `0.1673` |
| `lightgbm` | `0.031958s` | `0.012658s` | `0.9429` | `0.1652` |
| `xgboost` | `0.033780s` | `0.011209s` | `0.9453` | `0.1585` |
| `catboost` | `0.308426s` | `0.002152s` | `0.9429` | `0.1519` |
