# Training micro summary | gradient_boosting | regression

This summary focuses on ForestFire training phases so preprocessing and learner-fit costs can be inspected independently.

## Median measured time

- `fit_end_to_end` median: `0.189640s`
- `fit_from_table` median: `0.144175s`
- `table_build` median: `0.037248s`

## Median share of end-to-end training

- `fit_from_table` share: `83.4%`
- `table_build` share: `15.5%`

## Scaling from smallest to largest row count

- `fit_end_to_end` median growth ratio: `5929.09x`
- `fit_from_table` median growth ratio: `11936.80x`
- `table_build` median growth ratio: `298.74x`
