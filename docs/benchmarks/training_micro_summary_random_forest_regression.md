# Training micro summary | random_forest | regression

This summary focuses on ForestFire training phases so preprocessing and learner-fit costs can be inspected independently.

## Median measured time

- `fit_end_to_end` median: `3.954034s`
- `fit_from_table` median: `3.898044s`
- `table_build` median: `0.027324s`

## Median share of end-to-end training

- `fit_from_table` share: `99.4%`
- `table_build` share: `0.7%`

## Scaling from smallest to largest row count

- `fit_end_to_end` median growth ratio: `291.26x`
- `fit_from_table` median growth ratio: `294.87x`
- `table_build` median growth ratio: `295.34x`
