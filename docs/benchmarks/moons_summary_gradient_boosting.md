# make_moons summary | gradient_boosting

This benchmark uses sklearn.make_moons to compare probability surfaces on a two-dimensional non-linear classification task.

Probability plot: `moons_probabilities_gradient_boosting.png`

| backend | train | predict | accuracy | log loss |
| --- | ---: | ---: | ---: | ---: |
| `forestfire_cart` | `0.000973s` | `0.000258s` | `0.5000` | `0.6931` |
  note: No trees were added. The ensemble stayed at its base score, so the probability surface is constant.
| `forestfire_oblivious` | `0.003883s` | `0.000357s` | `0.9326` | `0.4157` |
| `sklearn` | `0.123548s` | `0.012538s` | `0.9473` | `0.1723` |
| `lightgbm` | `0.014927s` | `0.003374s` | `0.9453` | `0.1754` |
| `xgboost` | `0.012895s` | `0.000985s` | `0.9419` | `0.1760` |
| `catboost` | `0.185344s` | `0.001051s` | `0.9512` | `0.1640` |
