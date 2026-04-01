# Prediction micro summary | gradient_boosting | classification | trained on 1024 rows

This summary fixes training complexity and varies only prediction batch size so runtime bottlenecks are easier to isolate.

## Median measured time

- `predict_base` median: `0.000027s`
- `predict_compiled` median: `0.000002s`
- `predict_optimized` median: `0.000005s`

## Speedup over base semantic prediction

- `predict_compiled` median speedup: `11.21x`
- `predict_optimized` median speedup: `5.56x`

## Scaling from smallest to largest batch size
