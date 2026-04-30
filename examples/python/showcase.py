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


def show_canary_filter_policy() -> None:
    rng = np.random.default_rng(23)
    x = rng.normal(size=(4_000, 6))
    y = (
        1.8 * x[:, 0]
        - 0.9 * x[:, 1]
        + 0.4 * x[:, 2]
        + rng.normal(scale=0.8, size=x.shape[0])
        > 0.0
    ).astype(float)

    strict = train(
        x,
        y,
        task="classification",
        tree_type="cart",
        canaries=2,
    )
    top_3 = train(
        x,
        y,
        task="classification",
        tree_type="cart",
        canaries=2,
        filter=3,
    )
    top_5_percent = train(
        x,
        y,
        task="classification",
        tree_type="cart",
        canaries=2,
        filter=0.95,
    )

    print_section("Canary Filter Policy")
    print("strict root   ->", strict.tree_structure())
    print("top_3 root    ->", top_3.tree_structure())
    print("top_5pct root ->", top_5_percent.tree_structure())


def show_oblique_models() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(6_000, 6))
    y = (x[:, 0] + x[:, 1] > 0.5).astype(float)

    axis_model = train(
        x,
        y,
        algorithm="dt",
        task="classification",
        tree_type="cart",
        split_strategy="axis_aligned",
        canaries=2,
    )
    oblique_model = train(
        x,
        y,
        algorithm="dt",
        task="classification",
        tree_type="cart",
        split_strategy="oblique",
        canaries=2,
    )
    optimized = oblique_model.optimize_inference(physical_cores=1)

    print_section("Oblique Splits")
    print("axis root     ->", axis_model.tree_node(0, tree_index=0))
    print("oblique root  ->", oblique_model.tree_node(0, tree_index=0))
    print("optimized pred->", optimized.predict_proba(x[:4]).tolist())


def show_builder_strategies() -> None:
    rng = np.random.default_rng(31)
    x = rng.normal(size=(8_000, 6))
    y = (
        ((x[:, 0] > 0.5) & (x[:, 1] < -0.1))
        | ((x[:, 2] + x[:, 3]) > 0.6)
        | ((x[:, 4] - x[:, 5]) > 1.0)
    ).astype(float)

    print_section("Builder Strategies")
    greedy = train(
        x,
        y,
        task="classification",
        tree_type="cart",
        builder="greedy",
        canaries=0,
    )
    lookahead = train(
        x,
        y,
        task="classification",
        tree_type="cart",
        builder="lookahead",
        lookahead_depth=2,
        lookahead_top_k=4,
        lookahead_weight=0.5,
        canaries=0,
    )
    beam = train(
        x,
        y,
        task="classification",
        tree_type="cart",
        builder="beam",
        lookahead_depth=2,
        lookahead_top_k=4,
        lookahead_weight=0.5,
        beam_width=2,
        canaries=0,
    )
    optimal = train(
        x,
        y,
        task="classification",
        tree_type="cart",
        builder="optimal",
        max_depth=4,
        canaries=2,
        filter=0.95,
    )

    for builder, model in (
        ("greedy", greedy),
        ("lookahead", lookahead),
        ("beam", beam),
        ("optimal", optimal),
    ):
        print(f"{builder:>9} ->", model.tree_structure())


def show_optimal_builder() -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(4_000, 5))
    y = (((x[:, 0] > 0.2) & (x[:, 1] < -0.3)) | ((x[:, 2] + x[:, 3]) > 0.8)).astype(
        float
    )

    model = train(
        x,
        y,
        task="classification",
        tree_type="cart",
        builder="optimal",
        max_depth=4,
        canaries=2,
        filter=0.95,
    )

    print_section("Optimal Builder")
    print(model.tree_structure())


def show_missing_value_routing() -> None:
    x = np.array(
        [
            [0.0, np.nan, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, np.nan, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=float)

    heuristic = train(
        x,
        y,
        task="classification",
        tree_type="cart",
        missing_value_strategy="heuristic",
    )
    optimal = train(
        x,
        y,
        task="classification",
        tree_type="cart",
        missing_value_strategy={"f0": "heuristic", "f1": "optimal", "f2": "heuristic"},
    )
    optimized = optimal.optimize_inference(physical_cores=1, missing_features=[1])

    print_section("Missing Value Routing")
    print("heuristic root->", heuristic.tree_node(0, tree_index=0))
    print("optimal root  ->", optimal.tree_node(0, tree_index=0))
    print("optimized pred->", optimized.predict_proba(x[:4]).tolist())


def show_categorical_strategies() -> None:
    x = [
        ["red", "small", 1.2],
        ["red", "large", 0.7],
        ["blue", "small", 2.4],
        ["blue", "large", 2.0],
        ["green", "small", 0.4],
        ["green", "large", 0.2],
        ["red", "small", 1.0],
        ["blue", "large", 2.3],
    ]
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=float)

    print_section("Categorical Strategies")
    for strategy in ("dummy", "target", "fisher"):
        model = train(
            x,
            y,
            task="classification",
            tree_type="cart",
            categorical_strategy=strategy,
            categorical_features=[0, 1],
            target_smoothing=20.0,
        )
        print(f"{strategy:>8} ->", model.predict(x).tolist())


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


def show_random_forests() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(400, 5))
    y_reg = x[:, 0] * 2.0 - x[:, 1] + rng.normal(scale=0.3, size=400)
    y_clf = (x[:, 0] + x[:, 1] > 0.0).astype(float)

    rf_reg = train(
        x,
        y_reg,
        algorithm="rf",
        task="regression",
        n_trees=50,
        max_features="sqrt",
        compute_oob=True,
        seed=0,
    )
    rf_clf = train(
        x,
        y_clf,
        algorithm="rf",
        task="classification",
        n_trees=50,
        max_features="sqrt",
        compute_oob=True,
        seed=0,
    )

    print_section("Random Forests")
    print("rf_reg oob_score  ->", round(rf_reg.oob_score, 4))
    print("rf_reg preds[:4]  ->", np.round(rf_reg.predict(x[:4]), 4).tolist())
    print("rf_clf oob_score  ->", round(rf_clf.oob_score, 4))
    print("rf_clf preds[:4]  ->", rf_clf.predict(x[:4]).tolist())
    print(
        "rf_clf proba[:2]  ->",
        np.round(rf_clf.predict_proba(x[:2]), 4).tolist(),
    )


def show_gradient_boosting() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(600, 5))
    y_reg = x[:, 0] * 1.5 - x[:, 2] + rng.normal(scale=0.4, size=600)
    y_clf = (x[:, 0] - x[:, 1] > 0.2).astype(float)

    gbm_reg = train(
        x,
        y_reg,
        algorithm="gbm",
        task="regression",
        n_trees=80,
        learning_rate=0.1,
        canaries=0,
        seed=0,
    )
    gbm_clf = train(
        x,
        y_clf,
        algorithm="gbm",
        task="classification",
        n_trees=80,
        learning_rate=0.1,
        top_gradient_fraction=0.5,
        other_gradient_fraction=0.1,
        seed=0,
    )

    print_section("Gradient Boosting")
    print("gbm_reg tree_count ->", gbm_reg.tree_count)
    print("gbm_reg preds[:4]  ->", np.round(gbm_reg.predict(x[:4]), 4).tolist())
    print("gbm_clf tree_count ->", gbm_clf.tree_count)
    print("gbm_clf preds[:4]  ->", gbm_clf.predict(x[:4]).tolist())
    print(
        "gbm_clf proba[:2]  ->",
        np.round(gbm_clf.predict_proba(x[:2]), 4).tolist(),
    )


def show_sample_weights() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 3))
    y = x[:, 0] * 2.0 + rng.normal(scale=0.5, size=200)
    weights = rng.uniform(0.5, 2.0, size=200)

    unweighted = train(x, y, task="regression", tree_type="cart", canaries=0)
    weighted = train(
        x, y, task="regression", tree_type="cart", canaries=0, sample_weight=weights
    )
    rf_weighted = train(
        x,
        y,
        algorithm="rf",
        task="regression",
        n_trees=30,
        seed=0,
        sample_weight=weights,
    )
    gbm_weighted = train(
        x,
        y,
        algorithm="gbm",
        task="regression",
        n_trees=30,
        learning_rate=0.1,
        canaries=0,
        seed=0,
        sample_weight=weights,
    )

    print_section("Sample Weights")
    print(
        "unweighted preds[:3]  ->",
        np.round(unweighted.predict(x[:3]), 4).tolist(),
    )
    print(
        "weighted preds[:3]    ->",
        np.round(weighted.predict(x[:3]), 4).tolist(),
    )
    print(
        "rf_weighted preds[:3] ->",
        np.round(rf_weighted.predict(x[:3]), 4).tolist(),
    )
    print(
        "gbm_weighted preds[:3]->",
        np.round(gbm_weighted.predict(x[:3]), 4).tolist(),
    )


def show_multi_target_regression() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(300, 4))
    # Two target columns: one linear combination, one nonlinear proxy
    y = np.column_stack(
        [
            x[:, 0] * 2.0 - x[:, 1],
            x[:, 2] + x[:, 3] * 0.5,
        ]
    )

    model = train(x, y, task="regression", tree_type="cart", canaries=0)
    preds = model.predict(x[:4])

    print_section("Multi-Target Regression")
    print("y shape           ->", y.shape)
    print("preds shape       ->", preds.shape)
    print("preds[:4]         ->", np.round(preds, 4).tolist())

    # Compatible with sample_weight
    weights = rng.uniform(0.5, 2.0, size=300)
    weighted_mt = train(
        x, y, task="regression", tree_type="cart", canaries=0, sample_weight=weights
    )
    print(
        "weighted preds[:2] ->",
        np.round(weighted_mt.predict(x[:2]), 4).tolist(),
    )


def show_feature_importances() -> None:
    rng = np.random.default_rng(99)
    x = rng.normal(size=(500, 6))
    # Only features 0, 1, 2 are informative
    y = x[:, 0] * 3.0 - x[:, 1] * 1.5 + x[:, 2] * 0.8 + rng.normal(scale=0.5, size=500)

    dt = train(x, y, task="regression", tree_type="cart", canaries=0)
    rf = train(x, y, algorithm="rf", task="regression", n_trees=50, seed=0)
    gbm = train(
        x,
        y,
        algorithm="gbm",
        task="regression",
        n_trees=50,
        learning_rate=0.1,
        seed=0,
    )

    print_section("Feature Importances (MDI)")
    for label, model in (("dt ", dt), ("rf ", rf), ("gbm", gbm)):
        imp = np.round(model.feature_importances_, 4).tolist()
        print(f"{label} importances ->", imp)
        print("    top feature  ->", int(np.argmax(model.feature_importances_)))


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
    show_canary_filter_policy()
    show_oblique_models()
    show_builder_strategies()
    show_optimal_builder()
    show_missing_value_routing()
    show_categorical_strategies()
    show_serialization()
    show_optional_sparse_input()
    show_random_forests()
    show_gradient_boosting()
    show_sample_weights()
    show_multi_target_regression()
    show_feature_importances()


if __name__ == "__main__":
    main()
