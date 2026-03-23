# Prediction summary | random_forest | regression

This summary is generated from the measured benchmark grid rather than being handwritten commentary.

## Fastest backend by grid cell

- `xgboost` is fastest in 12 grid cell(s)
- `forestfire_optimized` is fastest in 8 grid cell(s)
- `sklearn` is fastest in 4 grid cell(s)

## Median measured time

- `forestfire` median: `4.419009s`
- `forestfire_optimized` median: `0.549380s`
- `lightgbm` median: `1.106018s`
- `sklearn` median: `0.570165s`
- `xgboost` median: `0.593162s`

## Scaling from smallest to largest row count

- `forestfire` median growth ratio: `247.41x`
- `forestfire_optimized` median growth ratio: `33.82x`
- `lightgbm` median growth ratio: `218.13x`
- `sklearn` median growth ratio: `48.02x`
- `xgboost` median growth ratio: `141.65x`

## ForestFire optimized-vs-base

- `forestfire_optimized` median speedup over base: `7.78x`
