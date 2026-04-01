# Prediction summary | gradient_boosting | regression

This summary is generated from the measured benchmark grid rather than being handwritten commentary.

## Fastest backend by grid cell

- `catboost` is fastest in 21 grid cell(s)
- `sklearn` is fastest in 3 grid cell(s)

## Median measured time

- `catboost` median: `0.031847s`
- `forestfire` median: `0.708759s`
- `forestfire_optimized` median: `0.187328s`
- `lightgbm` median: `0.200960s`
- `sklearn` median: `0.196760s`
- `xgboost` median: `0.091960s`

## Scaling from smallest to largest row count

- `catboost` median growth ratio: `41.47x`
- `forestfire` median growth ratio: `71.05x`
- `forestfire_optimized` median growth ratio: `30.26x`
- `lightgbm` median growth ratio: `50.57x`
- `sklearn` median growth ratio: `41.82x`
- `xgboost` median growth ratio: `47.62x`

## ForestFire optimized-vs-base

- `forestfire_optimized` median speedup over base: `5.09x`
