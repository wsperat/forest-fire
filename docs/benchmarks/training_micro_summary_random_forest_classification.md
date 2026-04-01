# Training micro summary | random_forest | classification

This summary focuses on ForestFire training phases so preprocessing and learner-fit costs can be inspected independently.

## Median measured time

- `fit_end_to_end` median: `7.432216s`
- `fit_from_table` median: `7.445301s`
- `table_build` median: `0.039249s`

## Median share of end-to-end training

- `fit_from_table` share: `99.8%`
- `table_build` share: `0.6%`

## Scaling from smallest to largest row count

- `fit_end_to_end` median growth ratio: `55.99x`
- `fit_from_table` median growth ratio: `50.75x`
- `table_build` median growth ratio: `291.73x`
