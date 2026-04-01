# Prediction micro summary | random_forest | regression | trained on 8192 rows

This summary fixes training complexity and varies only prediction batch size so runtime bottlenecks are easier to isolate.

## Median measured time

- `predict_base` median: `0.000557s`
- `predict_compiled` median: `0.000569s`
- `predict_optimized` median: `0.000606s`

## Speedup over base semantic prediction

- `predict_compiled` median speedup: `0.99x`
- `predict_optimized` median speedup: `0.92x`

## Scaling from smallest to largest batch size

- `predict_base` median growth ratio: `58.60x`
- `predict_compiled` median growth ratio: `287.84x`
- `predict_optimized` median growth ratio: `1278.87x`
