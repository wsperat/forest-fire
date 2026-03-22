# Python API

The Python surface is centered on:

- `Table`
- `train(...)`
- `Model`
- `OptimizedModel`

## Training

```python
train(
    x,
    y=None,
    algorithm="dt",
    task="auto",
    tree_type="cart",
    criterion="auto",
    canaries=2,
    bins="auto",
    physical_cores=None,
    max_depth=None,
    min_samples_split=None,
    min_samples_leaf=None,
    n_trees=None,
    max_features=None,
    seed=None,
    compute_oob=False,
    learning_rate=None,
    bootstrap=False,
    top_gradient_fraction=None,
    other_gradient_fraction=None,
)
```

## Supported input types

- NumPy arrays
- Python sequences
- pandas
- polars
- pyarrow
- SciPy dense matrices
- SciPy sparse matrices

Single-row prediction also accepts 1D inputs like:

- `[1, 2, 3]`
- `np.array([1, 2, 3])`

## Main model methods

- `predict(...)`
- `predict_proba(...)`
- `optimize_inference(...)`
- `serialize(...)`
- `to_ir_json(...)`
- `to_dataframe(...)`

## Introspection

- `tree_count`
- `tree_structure(...)`
- `tree_prediction_stats(...)`
- `tree_node(...)`
- `tree_level(...)`
- `tree_leaf(...)`

`to_dataframe(...)` returns a `polars.DataFrame` when `polars` is installed and falls back to a `pyarrow.Table` otherwise.
