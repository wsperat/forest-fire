# Training micro summary | random_forest | regression

This summary focuses on ForestFire training phases so preprocessing and learner-fit costs can be inspected independently.

## Median measured time

- `fit_end_to_end` median: `0.176595s`
- `fit_from_table` median: `0.127327s`
- `table_build` median: `0.044284s`

## Median share of end-to-end training

- `fit_from_table` share: `78.2%`
- `table_build` share: `25.1%`

## Scaling from smallest to largest row count

- `fit_end_to_end` median growth ratio: `3.85x`
- `fit_from_table` median growth ratio: `5.15x`
- `table_build` median growth ratio: `4.47x`
