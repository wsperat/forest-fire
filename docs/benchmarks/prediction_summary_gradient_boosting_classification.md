# Prediction summary | gradient_boosting | classification

This summary is generated from the measured benchmark grid rather than being handwritten commentary.

## Fastest backend by grid cell

- `lightgbm` is fastest in 8 grid cell(s)
- `xgboost` is fastest in 7 grid cell(s)
- `catboost` is fastest in 5 grid cell(s)
- `sklearn` is fastest in 4 grid cell(s)

## Median measured time

- `catboost` median: `0.031495s`
- `forestfire` median: `0.221012s`
- `forestfire_optimized` median: `0.145434s`
- `lightgbm` median: `0.035446s`
- `sklearn` median: `0.043941s`
- `xgboost` median: `0.024279s`

## Scaling from smallest to largest row count

- `catboost` median growth ratio: `45.09x`
- `forestfire` median growth ratio: `66.41x`
- `forestfire_optimized` median growth ratio: `44.76x`
- `lightgbm` median growth ratio: `40.54x`
- `sklearn` median growth ratio: `43.37x`
- `xgboost` median growth ratio: `38.04x`

## ForestFire optimized-vs-base

- `forestfire_optimized` median speedup over base: `1.72x`
