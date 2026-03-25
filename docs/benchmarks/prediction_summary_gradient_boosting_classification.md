# Prediction summary | gradient_boosting | classification

This summary is generated from the measured benchmark grid rather than being handwritten commentary.

## Fastest backend by grid cell

- `lightgbm` is fastest in 8 grid cell(s)
- `xgboost` is fastest in 7 grid cell(s)
- `sklearn` is fastest in 5 grid cell(s)
- `catboost` is fastest in 2 grid cell(s)
- `forestfire` is fastest in 2 grid cell(s)

## Median measured time

- `catboost` median: `0.056631s`
- `forestfire` median: `0.252805s`
- `forestfire_optimized` median: `0.160229s`
- `lightgbm` median: `0.032923s`
- `sklearn` median: `0.075586s`
- `xgboost` median: `0.025651s`

## Scaling from smallest to largest row count

- `catboost` median growth ratio: `49.16x`
- `forestfire` median growth ratio: `86.73x`
- `forestfire_optimized` median growth ratio: `58.72x`
- `lightgbm` median growth ratio: `54.25x`
- `sklearn` median growth ratio: `44.69x`
- `xgboost` median growth ratio: `46.56x`

## ForestFire optimized-vs-base

- `forestfire_optimized` median speedup over base: `1.49x`
