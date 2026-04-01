# Training micro summary | gradient_boosting | classification

This summary focuses on ForestFire training phases so preprocessing and learner-fit costs can be inspected independently.

## Median measured time

- `fit_end_to_end` median: `0.992477s`
- `fit_from_table` median: `0.999361s`
- `table_build` median: `0.039061s`

## Median share of end-to-end training

- `fit_from_table` share: `98.3%`
- `table_build` share: `2.5%`

## Scaling from smallest to largest row count

- `fit_end_to_end` median growth ratio: `67.66x`
- `fit_from_table` median growth ratio: `63.52x`
- `table_build` median growth ratio: `284.83x`
