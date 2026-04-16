# make_moons summary | gradient_boosting

This benchmark uses sklearn.make_moons to compare probability surfaces on a two-dimensional non-linear classification task.

Probability plot: `moons_probabilities_gradient_boosting.png`

| backend | train | predict | accuracy | log loss |
| --- | ---: | ---: | ---: | ---: |
| `forestfire_cart` | `0.003182s` | `0.001153s` | `0.5000` | `0.6931` |
  note: No trees were added. The ensemble stayed at its base score, so the probability surface is constant.
| `forestfire_oblivious` | `0.005469s` | `0.001301s` | `0.8888` | `0.5769` |
| `sklearn` | `0.133885s` | `0.020901s` | `0.9086` | `0.2355` |
| `lightgbm` | `0.030878s` | `0.014413s` | `0.9076` | `0.2369` |
| `xgboost` | `0.019859s` | `0.003688s` | `0.9072` | `0.2373` |
| `catboost` | `0.386568s` | `0.002678s` | `0.9129` | `0.2297` |
