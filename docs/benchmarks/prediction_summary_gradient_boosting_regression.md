# Prediction summary | gradient_boosting | regression

This summary is generated from the measured benchmark grid rather than being handwritten commentary.

## Fastest backend by grid cell

- `catboost` is fastest in 11 grid cell(s)
- `xgboost` is fastest in 4 grid cell(s)
- `forestfire` is fastest in 1 grid cell(s)
- `sklearn` is fastest in 1 grid cell(s)

## Median measured time

- `catboost` median: `0.007579s`
- `forestfire` median: `0.120736s`
- `forestfire_optimized` median: `0.048216s`
- `lightgbm` median: `0.012444s`
- `sklearn` median: `0.014896s`
- `xgboost` median: `0.008564s`

## Scaling from smallest to largest row count

- `catboost` median growth ratio: `21.46x`
- `forestfire` median growth ratio: `46.21x`
- `forestfire_optimized` median growth ratio: `15.43x`
- `lightgbm` median growth ratio: `51.36x`
- `sklearn` median growth ratio: `61.37x`
- `xgboost` median growth ratio: `37.22x`

## ForestFire optimized-vs-base

- `forestfire_optimized` median speedup over base: `1.36x`
