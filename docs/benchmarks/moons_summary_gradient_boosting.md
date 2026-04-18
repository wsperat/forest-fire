# make_moons summary | gradient_boosting

This benchmark uses sklearn.make_moons to compare probability surfaces on a two-dimensional non-linear classification task.

Probability plot: `moons_probabilities_gradient_boosting.png`

| backend | train | predict | accuracy | log loss |
| --- | ---: | ---: | ---: | ---: |
| `forestfire_cart` | `0.036952s` | `0.003507s` | `0.9092` | `0.2153` |
| `forestfire_randomized` | `0.059639s` | `0.004656s` | `0.9141` | `0.2204` |
| `forestfire_oblivious` | `0.057846s` | `0.001431s` | `0.9199` | `0.2176` |
| `sklearn` | `2.531784s` | `0.014047s` | `0.8965` | `0.2884` |
| `lightgbm` | `0.052428s` | `0.020786s` | `0.8916` | `0.2809` |
| `xgboost` | `0.093001s` | `0.016479s` | `0.9004` | `0.2445` |
| `catboost` | `0.421513s` | `0.002099s` | `0.8989` | `0.2451` |
