import importlib
import json
from importlib import util
from typing import Any

import numpy as np
from forestfire import Table, train


def print_section(title: str) -> None:
    print(f"\n== {title} ==")


def regression_rows() -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0], dtype=float)
    return x, y


def classification_rows() -> tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )
    y = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=float)
    return x, y


def show_regression_models() -> None:
    x, y = regression_rows()
    configs = [
        ("cart", "mean"),
        ("cart", "median"),
        ("randomized", "mean"),
        ("oblivious", "mean"),
    ]

    print_section("Regression Models")
    for tree_type, criterion in configs:
        model = train(
            x,
            y,
            task="regression",
            tree_type=tree_type,
            criterion=criterion,
            canaries=0,
            bins="auto",
        )
        preds = model.predict(x)
        print(f"{tree_type:>11} / {criterion:>6} -> {np.round(preds[:4], 4).tolist()}")


def show_classification_models() -> None:
    x, y = classification_rows()
    configs = [
        ("id3", "entropy"),
        ("c45", "entropy"),
        ("cart", "gini"),
        ("cart", "entropy"),
        ("randomized", "gini"),
        ("oblivious", "gini"),
    ]

    print_section("Classification Models")
    for tree_type, criterion in configs:
        model = train(
            x,
            y,
            task="classification",
            tree_type=tree_type,
            criterion=criterion,
            canaries=0,
            bins=64,
        )
        preds = model.predict(x)
        print(f"{tree_type:>11} / {criterion:>7} -> {preds.tolist()}")


def show_training_tables() -> None:
    x, y = classification_rows()
    dense_table = Table(np.column_stack([x[:, 0], x[:, 1] + 0.25]), y, bins="auto")
    sparse_x = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    sparse_table = Table(sparse_x, y, bins=8)

    print_section("Training Tables")
    print(
        "dense_table",
        {
            "kind": dense_table.kind,
            "n_rows": dense_table.n_rows,
            "n_features": dense_table.n_features,
            "canaries": dense_table.canaries,
        },
    )
    print(
        "sparse_table",
        {
            "kind": sparse_table.kind,
            "n_rows": sparse_table.n_rows,
            "n_features": sparse_table.n_features,
            "canaries": sparse_table.canaries,
        },
    )


def show_inference_inputs_and_optimized_runtime() -> None:
    x, y = classification_rows()
    model = train(
        x,
        y,
        task="classification",
        tree_type="cart",
        criterion="gini",
        canaries=0,
        bins="auto",
    )
    optimized = model.optimize_inference(physical_cores=1)

    named = {"f0": [0.0, 0.0, 1.0, 2.0], "f1": [0.0, 1.0, 1.0, 0.0]}
    rows = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]

    print_section("Inference Inputs")
    print("raw rows      ->", model.predict(rows).tolist())
    print("named columns ->", model.predict(named).tolist())
    print("optimized     ->", optimized.predict(rows).tolist())

    compiled = optimized.serialize_compiled()
    restored_optimized = type(optimized).deserialize_compiled(
        compiled, physical_cores=1
    )
    print("compiled size ->", len(compiled))
    print("compiled pred ->", restored_optimized.predict(rows).tolist())


def show_serialization() -> None:
    x, y = regression_rows()
    model = train(x, y, task="regression", tree_type="cart", canaries=0, bins="auto")
    serialized = model.serialize()
    restored = type(model).deserialize(serialized)
    ir = json.loads(model.to_ir_json(pretty=True))

    print_section("Serialization")
    print("json bytes    ->", len(serialized.encode("utf-8")))
    print("ir keys       ->", sorted(ir.keys()))
    print("restored pred ->", restored.predict(x[:3]).tolist())


def show_optional_sparse_input() -> None:
    print_section("Optional SciPy Sparse Input")
    try:
        scipy_sparse_module = util.find_spec("scipy.sparse")
    except ModuleNotFoundError:
        scipy_sparse_module = None
    if scipy_sparse_module is None:
        print("SciPy not installed; skipping sparse-input example.")
        return
    sp: Any = importlib.import_module("scipy.sparse")

    x_sparse = sp.csr_matrix(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    y = np.array([0.0, 1.0, 1.0, 0.0], dtype=float)

    model = train(
        x_sparse,
        y,
        task="classification",
        tree_type="cart",
        canaries=0,
    )
    print("sparse preds  ->", model.predict(x_sparse).tolist())


def main() -> None:
    show_regression_models()
    show_classification_models()
    show_training_tables()
    show_inference_inputs_and_optimized_runtime()
    show_serialization()
    show_optional_sparse_input()


if __name__ == "__main__":
    main()
