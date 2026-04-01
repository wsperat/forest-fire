# Prediction summary | random_forest | regression

This summary is generated from the measured benchmark grid rather than being handwritten commentary.

## Fastest backend by grid cell

- `forestfire_optimized` is fastest in 11 grid cell(s)
- `sklearn` is fastest in 9 grid cell(s)
- `xgboost` is fastest in 4 grid cell(s)

## Median measured time

- `forestfire` median: `4.661428s`
- `forestfire_optimized` median: `0.518340s`
- `lightgbm` median: `2.380745s`
- `sklearn` median: `0.618993s`
- `xgboost` median: `0.957352s`

## Scaling from smallest to largest row count

- `forestfire` median growth ratio: `69.47x`
- `forestfire_optimized` median growth ratio: `18.72x`
- `lightgbm` median growth ratio: `58.89x`
- `sklearn` median growth ratio: `35.47x`
- `xgboost` median growth ratio: `55.36x`

## ForestFire optimized-vs-base

- `forestfire_optimized` median speedup over base: `10.99x`
