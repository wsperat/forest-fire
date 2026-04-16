# make_moons summary | random_forest

This benchmark uses sklearn.make_moons to compare probability surfaces on a two-dimensional non-linear classification task.

Probability plot: `moons_probabilities_random_forest.png`

| backend | train | predict | accuracy | log loss |
| --- | ---: | ---: | ---: | ---: |
| `forestfire_cart` | `3.156506s` | `0.045716s` | `0.8940` | `0.5709` |
| `forestfire_randomized` | `3.921815s` | `0.049553s` | `0.9033` | `0.2418` |
| `forestfire_oblivious` | `1.181687s` | `0.019111s` | `0.9136` | `0.2192` |
| `sklearn` | `0.803524s` | `0.026551s` | `0.9038` | `0.5455` |
| `lightgbm` | `0.579494s` | `0.039765s` | `0.8853` | `0.3140` |
| `xgboost` | `0.366545s` | `0.015468s` | `0.9048` | `0.5067` |
