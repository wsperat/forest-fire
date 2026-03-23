# Prediction summary | random_forest | classification

This summary is generated from the measured benchmark grid rather than being handwritten commentary.

## Fastest backend by grid cell

- `xgboost` is fastest in 18 grid cell(s)
- `sklearn` is fastest in 6 grid cell(s)

## Median measured time

- `forestfire` median: `4.757934s`
- `forestfire_optimized` median: `3.023567s`
- `lightgbm` median: `0.618503s`
- `sklearn` median: `0.284330s`
- `xgboost` median: `0.185379s`

## Scaling from smallest to largest row count

- `forestfire` median growth ratio: `110.31x`
- `forestfire_optimized` median growth ratio: `72.33x`
- `lightgbm` median growth ratio: `106.67x`
- `sklearn` median growth ratio: `15.91x`
- `xgboost` median growth ratio: `100.49x`

## ForestFire optimized-vs-base

- `forestfire_optimized` median speedup over base: `1.75x`
