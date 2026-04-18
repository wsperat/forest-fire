# Prediction summary | random_forest | classification

This summary is generated from the measured benchmark grid rather than being handwritten commentary.

## Fastest backend by grid cell

- `xgboost` is fastest in 18 grid cell(s)
- `sklearn` is fastest in 6 grid cell(s)

## Median measured time

- `forestfire` median: `5.339357s`
- `forestfire_optimized` median: `3.286692s`
- `lightgbm` median: `0.641861s`
- `sklearn` median: `0.316377s`
- `xgboost` median: `0.248507s`

## Scaling from smallest to largest row count

- `forestfire` median growth ratio: `68.35x`
- `forestfire_optimized` median growth ratio: `66.96x`
- `lightgbm` median growth ratio: `56.51x`
- `sklearn` median growth ratio: `15.90x`
- `xgboost` median growth ratio: `54.07x`

## ForestFire optimized-vs-base

- `forestfire_optimized` median speedup over base: `1.77x`
