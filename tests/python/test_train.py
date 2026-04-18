import json

import numpy as np
import pytest
from forestfire import OptimizedModel, Table, train
from numpy.typing import NDArray

PREDICTION_TOLERANCE = 10e-6


@pytest.fixture
def toy_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([10.0, 12.0, 14.0, 20.0])
    return X, y


@pytest.fixture
def and_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0.0, 0.0, 0.0, 1.0])
    return X, y


def test_train_defaults_match_documented_regression_baseline(
    toy_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = toy_data

    model = train(X, y)

    assert model.algorithm == "dt"
    assert model.task == "regression"
    assert model.tree_type == "cart"
    assert model.criterion == "mean"


def test_table_accepts_integer_numpy_targets_without_crashing() -> None:
    X = np.random.randn(128, 3)
    y = np.random.choice([1, 0], 128)

    table = Table(X, y)

    assert table.n_rows == 128


def test_train_auto_detects_integer_targets_as_classification() -> None:
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0, 0, 1, 1], dtype=np.int64)

    model = train(X, y, tree_type="cart", canaries=0)

    assert model.task == "classification"
    assert np.array_equal(model.predict(X), y.astype(np.float64))


def test_train_auto_defaults_to_cart_for_classification_targets() -> None:
    X = np.random.randn(100, 3)
    y = np.random.choice([0, 1], 100)

    model = train(X, y)

    assert model.task == "classification"
    assert model.tree_type == "cart"


def test_train_auto_detects_float_targets_as_regression() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)

    model = train(X, y, tree_type="cart", canaries=0)

    assert model.task == "regression"
    assert model.tree_type == "cart"


def test_train_auto_detects_string_targets_as_classification() -> None:
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array(["cat", "cat", "dog", "dog"])

    model = train(X, y, tree_type="cart", canaries=0)
    preds = model.predict(X)

    assert model.task == "classification"
    assert preds.tolist() == y.tolist()


def test_table_builds_sparse_layout_for_binary_data() -> None:
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0])

    table = Table(X, y, canaries=1)

    assert table.kind == "sparse"
    assert table.n_rows == 3
    assert table.n_features == 2
    assert table.canaries == 1


def test_table_builds_dense_layout_for_mixed_data() -> None:
    X = np.array([[0.0, 1.5], [1.0, 0.0], [1.0, 2.0]])
    y = np.array([0.0, 1.0, 1.0])

    table = Table(X, y, canaries=1)

    assert table.kind == "dense"
    assert table.n_rows == 3
    assert table.n_features == 2


def test_table_accepts_auto_and_fixed_bins() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])

    auto_table = Table(X, y, bins="auto")
    fixed_table = Table(X, y, bins=64)

    assert auto_table.kind == "dense"
    assert fixed_table.kind == "dense"


@pytest.mark.parametrize("bins", [0, 129])
def test_table_rejects_invalid_integer_bins(bins: int) -> None:
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="between 1 and 128"):
        Table(X, y, bins=bins)


def test_table_rejects_invalid_string_bins() -> None:
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])

    with pytest.raises(
        ValueError,
        match="Expected 'auto' or an integer between 1 and 128",
    ):
        Table(X, y, bins="dynamic")


def test_train_accepts_integer_filter_argument(
    toy_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = toy_data

    model = train(X, y, canaries=0, filter=3)

    preds = model.predict(X)

    assert preds.shape == y.shape
    assert np.all(np.isfinite(preds))


def test_train_accepts_fractional_filter_argument(
    toy_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = toy_data

    model = train(X, y, canaries=0, filter=0.5)

    preds = model.predict(X)

    assert preds.shape == y.shape
    assert np.all(np.isfinite(preds))


@pytest.mark.parametrize("value", [0, -1, 1.0, -0.1, 1.1])
def test_train_rejects_invalid_filter_values(value: int | float) -> None:
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="filter"):
        train(X, y, filter=value)


def test_train_accepts_prebuilt_table(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    table = Table(X, y, canaries=0)

    model = train(table, task="classification", tree_type="cart")

    assert model.algorithm == "dt"
    assert np.array_equal(model.predict(table), y)


def test_train_raises_on_mismatched_lengths() -> None:
    X = np.zeros((3, 1))
    y = np.ones(2)
    with pytest.raises(ValueError):
        train(X, y)


def test_train_cart_classifier(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(
        X, y, algorithm="dt", task="classification", tree_type="cart", canaries=0
    )

    assert model.algorithm == "dt"
    assert model.task == "classification"
    assert model.criterion == "gini"
    assert model.tree_type == "cart"
    assert np.array_equal(model.predict(X), y)


def test_predict_proba_returns_class_probabilities(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, _y = and_data
    model = train(
        X,
        np.array([0.0, 0.0, 0.0, 1.0]),
        task="classification",
        tree_type="cart",
        canaries=0,
    )

    proba = model.predict_proba(X)

    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(
        proba, np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    )


def test_predict_proba_batch_and_single_row_match(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(X, y, task="classification", tree_type="cart", canaries=0)

    batch_proba = model.predict_proba(X)
    single_row_proba = np.vstack(
        [model.predict_proba(X[row_idx : row_idx + 1]) for row_idx in range(X.shape[0])]
    )

    assert np.allclose(
        batch_proba,
        single_row_proba,
        atol=PREDICTION_TOLERANCE,
        rtol=PREDICTION_TOLERANCE,
    )


def test_predict_proba_accepts_named_feature_dict(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(X, y, task="classification", tree_type="cart", canaries=0)

    proba = model.predict_proba(
        {"f0": [0.0, 0.0, 1.0, 1.0], "f1": [0.0, 1.0, 0.0, 1.0]}
    )

    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(np.argmax(proba, axis=1), y.astype(int))


def test_predict_proba_accepts_single_named_feature_row(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(X, y, task="classification", tree_type="cart", canaries=0)

    proba = model.predict_proba({"f0": 1.0, "f1": 1.0})

    assert proba.shape == (1, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.argmax(proba[0]) == int(y[-1])


def test_predict_proba_rejects_regression_models(
    toy_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = toy_data
    model = train(X, y, task="regression", tree_type="cart", canaries=0)

    with pytest.raises(ValueError, match="only available for classification models"):
        model.predict_proba(X)


def test_train_random_forest_classifier_is_serialized_as_an_ensemble(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    forest = train(
        X,
        y,
        algorithm="rf",
        task="classification",
        tree_type="cart",
        canaries=0,
        n_trees=3,
    )

    assert forest.algorithm == "rf"
    assert forest.task == "classification"
    assert forest.criterion == "gini"
    assert forest.tree_type == "cart"
    assert forest.canaries == 0
    assert forest.max_depth == 8
    assert forest.min_samples_split == 2
    assert forest.min_samples_leaf == 1
    assert forest.n_trees == 3
    assert forest.max_features == 1
    assert forest.seed is None
    assert np.array_equal(forest.predict(X), y)
    assert np.allclose(forest.predict_proba(X).sum(axis=1), 1.0)

    serialized = json.loads(forest.serialize())
    assert serialized["model"]["is_ensemble"] is True
    assert len(serialized["model"]["trees"]) == 3


def test_train_random_forest_regressor_bootstrap_averages_predictions() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 0.0, 100.0])
    forest = train(
        X,
        y,
        algorithm="rf",
        task="regression",
        tree_type="cart",
        canaries=0,
        n_trees=5,
    )
    tree = train(X, y, algorithm="dt", task="regression", tree_type="cart", canaries=0)
    forest_preds = forest.predict(X)
    tree_preds = tree.predict(X)

    assert forest.algorithm == "rf"
    assert forest.task == "regression"
    assert forest.criterion == "mean"
    assert forest.tree_type == "cart"
    assert forest_preds.shape == tree_preds.shape
    assert np.all(np.isfinite(forest_preds))
    assert not np.allclose(forest_preds, tree_preds)


def test_train_random_forest_classifier_bootstrap_changes_probability_estimates() -> (
    None
):
    X = np.array([[0.0], [0.0], [0.0], [1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    forest = train(
        X,
        y,
        algorithm="rf",
        task="classification",
        tree_type="cart",
        canaries=0,
        n_trees=5,
    )
    tree = train(
        X, y, algorithm="dt", task="classification", tree_type="cart", canaries=0
    )

    forest_proba = forest.predict_proba(X)
    tree_proba = tree.predict_proba(X)

    assert np.allclose(forest_proba.sum(axis=1), 1.0)
    assert not np.allclose(forest_proba, tree_proba)


def test_random_forest_rejects_zero_trees(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data

    with pytest.raises(ValueError, match="requires at least one tree"):
        train(
            X,
            y,
            algorithm="rf",
            task="classification",
            tree_type="cart",
            canaries=0,
            n_trees=0,
        )


def test_random_forest_round_trips_through_serialization(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(
        X,
        y,
        algorithm="rf",
        task="classification",
        tree_type="cart",
        canaries=0,
        n_trees=3,
    )
    restored = model.deserialize(model.serialize())

    assert restored.algorithm == "rf"
    assert np.array_equal(restored.predict(X), model.predict(X))
    assert np.allclose(restored.predict_proba(X), model.predict_proba(X))


def test_model_and_optimized_model_expose_hyperparameters(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(
        X,
        y,
        algorithm="dt",
        task="classification",
        tree_type="cart",
        canaries=0,
        max_depth=4,
        min_samples_split=3,
        min_samples_leaf=2,
    )
    optimized = model.optimize_inference()

    assert model.canaries == 0
    assert model.max_depth == 4
    assert model.min_samples_split == 3
    assert model.min_samples_leaf == 2
    assert model.n_trees is None
    assert model.max_features is None
    assert model.seed is None
    assert model.compute_oob is False
    assert model.oob_score is None

    assert optimized.canaries == model.canaries
    assert optimized.max_depth == model.max_depth
    assert optimized.min_samples_split == model.min_samples_split
    assert optimized.min_samples_leaf == model.min_samples_leaf
    assert optimized.n_trees == model.n_trees
    assert optimized.max_features == model.max_features
    assert optimized.seed == model.seed
    assert optimized.compute_oob == model.compute_oob
    assert optimized.oob_score == model.oob_score


def test_train_accepts_min_samples_hyperparameters(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(
        X,
        y,
        task="classification",
        tree_type="cart",
        canaries=0,
        min_samples_split=5,
        min_samples_leaf=2,
    )

    assert model.min_samples_split == 5
    assert model.min_samples_leaf == 2


def test_train_accepts_max_depth_hyperparameter(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(
        X,
        y,
        task="classification",
        tree_type="cart",
        canaries=0,
        max_depth=3,
    )

    assert model.max_depth == 3


def test_train_rejects_zero_min_samples_values(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data

    with pytest.raises(ValueError, match="min_samples_split must be at least 1"):
        train(X, y, min_samples_split=0)

    with pytest.raises(ValueError, match="min_samples_leaf must be at least 1"):
        train(X, y, min_samples_leaf=0)

    with pytest.raises(ValueError, match="max_depth must be at least 1"):
        train(X, y, max_depth=0)


def test_random_forest_defaults_to_1000_trees(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    forest = train(
        X,
        y,
        algorithm="rf",
        task="classification",
        tree_type="cart",
        canaries=0,
    )

    assert forest.n_trees == 1000


def test_random_forest_classifier_exposes_oob_score_when_requested() -> None:
    rng = np.random.default_rng(7)
    X = rng.standard_normal((1_000, 4))
    y = rng.integers(0, 2, size=1_000)
    model = train(
        X,
        y,
        algorithm="rf",
        task="classification",
        tree_type="cart",
        canaries=0,
        n_trees=50,
        seed=13,
        compute_oob=True,
    )
    optimized = model.optimize_inference(physical_cores=1)

    assert model.compute_oob is True
    assert model.oob_score is not None
    assert 0.0 <= model.oob_score <= 1.0
    assert optimized.compute_oob is True
    assert optimized.oob_score == model.oob_score


def test_random_forest_regressor_exposes_oob_score_when_requested() -> None:
    rng = np.random.default_rng(7)
    X = rng.standard_normal((1_000, 4))
    y = X[:, 0] * 2.0 - X[:, 1] + rng.normal(scale=0.1, size=1_000)
    model = train(
        X,
        y,
        algorithm="rf",
        task="regression",
        tree_type="cart",
        canaries=0,
        n_trees=50,
        seed=13,
        compute_oob=True,
    )

    assert model.compute_oob is True
    assert model.oob_score is not None
    assert np.isfinite(model.oob_score)


@pytest.mark.parametrize("tree_type", ["id3", "c45", "cart", "randomized", "oblivious"])
def test_classification_tree_max_depth_affects_learning(tree_type: str) -> None:
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    shallow = train(
        X, y, task="classification", tree_type=tree_type, canaries=0, max_depth=1
    )
    deep = train(
        X, y, task="classification", tree_type=tree_type, canaries=0, max_depth=8
    )

    assert not np.array_equal(shallow.predict(X), deep.predict(X))


@pytest.mark.parametrize("tree_type", ["id3", "c45", "cart", "randomized", "oblivious"])
def test_classification_tree_min_samples_split_affects_learning(tree_type: str) -> None:
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    constrained = train(
        X,
        y,
        task="classification",
        tree_type=tree_type,
        canaries=0,
        min_samples_split=len(X) + 1,
    )
    baseline = train(X, y, task="classification", tree_type=tree_type, canaries=0)

    assert np.unique(constrained.predict(X)).size == 1
    assert not np.array_equal(constrained.predict(X), baseline.predict(X))


@pytest.mark.parametrize("tree_type", ["id3", "c45", "cart", "randomized", "oblivious"])
def test_classification_tree_min_samples_leaf_affects_learning(tree_type: str) -> None:
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    constrained = train(
        X,
        y,
        task="classification",
        tree_type=tree_type,
        canaries=0,
        min_samples_leaf=5,
    )
    baseline = train(X, y, task="classification", tree_type=tree_type, canaries=0)

    assert np.unique(constrained.predict(X)).size == 1
    assert not np.array_equal(constrained.predict(X), baseline.predict(X))


@pytest.mark.parametrize("tree_type", ["cart", "randomized", "oblivious"])
def test_regression_tree_max_depth_affects_learning(tree_type: str) -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0])

    shallow = train(
        X, y, task="regression", tree_type=tree_type, canaries=0, max_depth=1
    )
    deep = train(X, y, task="regression", tree_type=tree_type, canaries=0, max_depth=8)

    assert not np.allclose(shallow.predict(X), deep.predict(X))


@pytest.mark.parametrize("tree_type", ["cart", "randomized", "oblivious"])
def test_regression_tree_min_samples_split_affects_learning(tree_type: str) -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0])

    constrained = train(
        X,
        y,
        task="regression",
        tree_type=tree_type,
        canaries=0,
        min_samples_split=len(X) + 1,
    )
    baseline = train(X, y, task="regression", tree_type=tree_type, canaries=0)

    assert np.unique(constrained.predict(X)).size == 1
    assert not np.allclose(constrained.predict(X), baseline.predict(X))


@pytest.mark.parametrize("tree_type", ["cart", "randomized", "oblivious"])
def test_regression_tree_min_samples_leaf_affects_learning(tree_type: str) -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0])

    constrained = train(
        X,
        y,
        task="regression",
        tree_type=tree_type,
        canaries=0,
        min_samples_leaf=5,
    )
    baseline = train(X, y, task="regression", tree_type=tree_type, canaries=0)

    assert np.unique(constrained.predict(X)).size == 1
    assert not np.allclose(constrained.predict(X), baseline.predict(X))


def test_random_forest_ignores_canaries(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    forest = train(
        X,
        y,
        algorithm="rf",
        task="classification",
        tree_type="cart",
        canaries=5,
        n_trees=3,
    )

    assert forest.canaries == 0
    serialized = json.loads(forest.serialize())
    assert serialized["training_metadata"]["canaries"] == 0


def test_random_forest_seed_makes_training_deterministic() -> None:
    X = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])

    left = train(
        X,
        y,
        algorithm="rf",
        task="classification",
        tree_type="cart",
        canaries=0,
        n_trees=7,
        max_features="all",
        seed=17,
    )
    right = train(
        X,
        y,
        algorithm="rf",
        task="classification",
        tree_type="cart",
        canaries=0,
        n_trees=7,
        max_features="all",
        seed=17,
    )

    assert np.array_equal(left.predict(X), right.predict(X))
    assert np.allclose(left.predict_proba(X), right.predict_proba(X))


def test_random_forest_seed_changes_probability_estimates() -> None:
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])

    left = train(
        X,
        y,
        algorithm="rf",
        task="classification",
        tree_type="cart",
        canaries=0,
        n_trees=9,
        max_features=1,
        seed=7,
    )
    right = train(
        X,
        y,
        algorithm="rf",
        task="classification",
        tree_type="cart",
        canaries=0,
        n_trees=9,
        max_features=1,
        seed=8,
    )

    assert np.allclose(left.predict_proba(X).sum(axis=1), 1.0)
    assert np.allclose(right.predict_proba(X).sum(axis=1), 1.0)
    assert not np.allclose(left.predict_proba(X), right.predict_proba(X))


def test_random_forest_rejects_zero_max_features(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data

    with pytest.raises(ValueError, match="max_features must be at least 1"):
        train(
            X,
            y,
            algorithm="rf",
            task="classification",
            tree_type="cart",
            canaries=0,
            n_trees=3,
            max_features=0,
        )


def test_predict_accepts_feature_only_table(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(X, y, task="classification", tree_type="cart", canaries=0)

    feature_table = Table(X)

    assert feature_table.kind == "sparse"
    assert np.array_equal(model.predict(feature_table), y)


def test_predict_accepts_named_feature_dict(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(X, y, task="classification", tree_type="cart", canaries=0)

    preds = model.predict({"f0": [0.0, 0.0, 1.0, 1.0], "f1": [0.0, 1.0, 0.0, 1.0]})

    assert np.array_equal(preds, y)


def test_predict_accepts_single_named_feature_row(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(X, y, task="classification", tree_type="cart", canaries=0)

    pred = model.predict({"f0": 1.0, "f1": 1.0})

    assert pred.shape == (1,)
    assert pred[0] == 1.0


def test_predict_rejects_missing_named_feature(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(X, y, task="classification", tree_type="cart", canaries=0)

    with pytest.raises(ValueError, match="Missing required feature 'f1'"):
        model.predict({"f0": [0.0, 1.0]})


def test_train_and_predict_handle_python_none_missing_values() -> None:
    X = [[0.0], [1.0], [None], [None]]
    y = np.array([0.0, 1.0, 0.0, 0.0])

    model = train(X, y, task="classification", tree_type="cart", canaries=0)

    preds = model.predict([[None], [0.0], [1.0]])

    assert np.array_equal(preds, np.array([0.0, 0.0, 1.0]))


def test_train_accepts_string_missing_value_strategy() -> None:
    X = [[0.0], [1.0], [None], [None]]
    y = np.array([0.0, 1.0, 0.0, 0.0])

    heuristic = train(
        X,
        y,
        task="classification",
        tree_type="cart",
        canaries=0,
        missing_value_strategy="heuristic",
    )
    optimal = train(
        X,
        y,
        task="classification",
        tree_type="cart",
        canaries=0,
        missing_value_strategy="optimal",
    )

    assert np.array_equal(
        heuristic.predict([[None], [0.0], [1.0]]), np.array([0.0, 0.0, 1.0])
    )
    assert np.array_equal(
        optimal.predict([[None], [0.0], [1.0]]), np.array([0.0, 0.0, 1.0])
    )


def test_train_accepts_per_column_missing_value_strategy_dict() -> None:
    X = [
        [0.0, 0.0],
        [1.0, 0.0],
        [None, 1.0],
        [None, 1.0],
    ]
    y = np.array([0.0, 1.0, 0.0, 0.0])

    model = train(
        X,
        y,
        task="classification",
        tree_type="cart",
        canaries=0,
        missing_value_strategy={"col_1": "optimal", "f1": "heuristic"},
    )

    assert np.array_equal(
        model.predict([[None, 1.0], [1.0, 0.0]]), np.array([0.0, 1.0])
    )


def test_train_rejects_invalid_missing_value_strategy_string() -> None:
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="Unsupported missing_value_strategy"):
        train(
            X,
            y,
            task="classification",
            tree_type="cart",
            canaries=0,
            missing_value_strategy="fast",
        )


def test_train_rejects_invalid_missing_value_strategy_dict_key() -> None:
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="Invalid missing_value_strategy feature key"):
        train(
            X,
            y,
            task="classification",
            tree_type="cart",
            canaries=0,
            missing_value_strategy={"feature_a": "optimal"},
        )


def test_train_rejects_invalid_missing_value_strategy_dict_value() -> None:
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="Unsupported missing_value_strategy"):
        train(
            X,
            y,
            task="classification",
            tree_type="cart",
            canaries=0,
            missing_value_strategy={"col_1": "fast"},
        )


def test_predict_rejects_unexpected_named_feature(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(X, y, task="classification", tree_type="cart", canaries=0)

    with pytest.raises(ValueError, match="Unexpected feature 'f2'"):
        model.predict({"f0": [0.0, 1.0], "f1": [0.0, 1.0], "f2": [0.0, 1.0]})


def test_predict_accepts_plain_python_rows(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(X, y, task="classification", tree_type="cart", canaries=0)

    preds = model.predict([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    assert np.array_equal(preds, np.array([0.0, 0.0, 1.0]))


def test_optimize_inference_preserves_predictions_and_ir(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    X_with_unused = np.column_stack([X, np.full((X.shape[0], 2), [10.0, 20.0])])
    model = train(
        X_with_unused,
        y,
        task="classification",
        tree_type="cart",
        canaries=0,
    )
    optimized = model.optimize_inference(physical_cores=1)

    assert optimized.algorithm == model.algorithm
    assert optimized.task == model.task
    assert optimized.criterion == model.criterion
    assert optimized.tree_type == model.tree_type
    assert optimized.serialize() == model.serialize()
    assert optimized.to_ir_json() == model.to_ir_json()
    assert model.used_feature_indices == [0, 1]
    assert optimized.used_feature_indices == [0, 1]
    assert optimized.used_feature_count == 2
    assert np.allclose(
        optimized.predict(X_with_unused),
        model.predict(X_with_unused),
        atol=PREDICTION_TOLERANCE,
        rtol=PREDICTION_TOLERANCE,
    )


def test_optimize_inference_missing_features_option_changes_missing_checks() -> None:
    X = np.array([[0.0], [0.0], [1.0]])
    y = np.array([0.0, 0.0, 1.0])

    model = train(X, y, task="classification", tree_type="cart", canaries=0)
    missing_aware = model.optimize_inference(physical_cores=1)
    missing_disabled = model.optimize_inference(physical_cores=1, missing_features=[])

    assert np.array_equal(missing_aware.predict([[np.nan]]), np.array([0.0]))
    assert not np.array_equal(
        missing_aware.predict([[np.nan]]),
        missing_disabled.predict([[np.nan]]),
    )


def test_optimized_inference_batch_and_single_row_predictions_match(
    toy_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = toy_data
    model = train(X, y, task="regression", tree_type="cart", canaries=0)
    optimized = model.optimize_inference(physical_cores=1)

    batch_preds = optimized.predict(X)
    single_row_preds = np.array(
        [
            optimized.predict(X[row_idx : row_idx + 1])[0]
            for row_idx in range(X.shape[0])
        ]
    )

    assert np.allclose(
        batch_preds,
        single_row_preds,
        atol=PREDICTION_TOLERANCE,
        rtol=PREDICTION_TOLERANCE,
    )
    assert np.allclose(
        batch_preds,
        model.predict(X),
        atol=PREDICTION_TOLERANCE,
        rtol=PREDICTION_TOLERANCE,
    )


def test_optimized_inference_accepts_named_feature_dict(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(X, y, task="classification", tree_type="cart", canaries=0)
    optimized = model.optimize_inference(physical_cores=1)

    preds = optimized.predict({"f0": [0.0, 0.0, 1.0, 1.0], "f1": [0.0, 1.0, 0.0, 1.0]})

    assert np.array_equal(preds, y)


def test_optimized_predict_proba_matches_base_model(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(X, y, task="classification", tree_type="cart", canaries=0)
    optimized = model.optimize_inference(physical_cores=1)

    assert np.allclose(optimized.predict_proba(X), model.predict_proba(X))


def test_optimized_predict_proba_batch_and_single_row_match(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(X, y, task="classification", tree_type="cart", canaries=0)
    optimized = model.optimize_inference(physical_cores=1)

    batch_proba = optimized.predict_proba(X)
    single_row_proba = np.vstack(
        [
            optimized.predict_proba(X[row_idx : row_idx + 1])
            for row_idx in range(X.shape[0])
        ]
    )

    assert np.allclose(
        batch_proba,
        single_row_proba,
        atol=PREDICTION_TOLERANCE,
        rtol=PREDICTION_TOLERANCE,
    )
    assert np.array_equal(np.argmax(batch_proba, axis=1), y.astype(int))


def test_optimized_predict_proba_accepts_named_feature_dict(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(X, y, task="classification", tree_type="cart", canaries=0)
    optimized = model.optimize_inference(physical_cores=1)

    proba = optimized.predict_proba(
        {"f0": [0.0, 0.0, 1.0, 1.0], "f1": [0.0, 1.0, 0.0, 1.0]}
    )

    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(np.argmax(proba, axis=1), y.astype(int))


def test_predict_accepts_single_row_python_list() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0])
    model = train(X, y, task="classification", tree_type="cart", canaries=0)

    pred = model.predict([1.0, 1.0])
    proba = model.predict_proba([1.0, 1.0])

    assert pred.shape == (1,)
    assert np.array_equal(pred, np.array([1.0]))
    assert proba.shape == (1, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_predict_accepts_single_row_numpy_vector() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0])
    model = train(X, y, task="classification", tree_type="cart", canaries=0)

    pred = model.predict(np.array([1.0, 1.0]))
    proba = model.predict_proba(np.array([1.0, 1.0]))

    assert pred.shape == (1,)
    assert np.array_equal(pred, np.array([1.0]))
    assert proba.shape == (1, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_optimized_predict_accepts_single_row_python_list() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0])
    model = train(X, y, task="classification", tree_type="cart", canaries=0)
    optimized = model.optimize_inference(physical_cores=1)

    pred = optimized.predict([1.0, 1.0])
    proba = optimized.predict_proba([1.0, 1.0])

    assert pred.shape == (1,)
    assert np.array_equal(pred, np.array([1.0]))
    assert proba.shape == (1, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_optimized_predict_accepts_single_row_numpy_vector() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0])
    model = train(X, y, task="classification", tree_type="cart", canaries=0)
    optimized = model.optimize_inference(physical_cores=1)

    pred = optimized.predict(np.array([1.0, 1.0]))
    proba = optimized.predict_proba(np.array([1.0, 1.0]))

    assert pred.shape == (1,)
    assert np.array_equal(pred, np.array([1.0]))
    assert proba.shape == (1, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_optimize_inference_rejects_zero_physical_cores() -> None:
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])
    model = train(X, y)

    with pytest.raises(ValueError, match="Requested 0 physical cores"):
        model.optimize_inference(physical_cores=0)


def test_random_forest_optimized_inference_matches_base_model(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(
        X,
        y,
        algorithm="rf",
        task="classification",
        tree_type="cart",
        canaries=0,
        n_trees=3,
    )
    optimized = model.optimize_inference(physical_cores=1)

    assert np.array_equal(optimized.predict(X), model.predict(X))
    assert np.allclose(optimized.predict_proba(X), model.predict_proba(X))


def test_random_forest_optimized_regression_matches_base_model() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([0.0, 1.0, 1.5, 3.0, 8.0])
    model = train(
        X,
        y,
        algorithm="rf",
        task="regression",
        tree_type="cart",
        canaries=0,
        n_trees=5,
        seed=7,
    )
    optimized = model.optimize_inference(physical_cores=1)

    assert np.allclose(optimized.predict(X), model.predict(X))


@pytest.mark.parametrize("tree_type", ["id3", "c45", "cart", "randomized", "oblivious"])
def test_optimized_classifier_predict_proba_matches_base_model_for_all_tree_types(
    tree_type: str,
) -> None:
    rng = np.random.default_rng(7)
    X = rng.standard_normal((2_000, 3))
    y = rng.integers(0, 2, size=2_000)
    model = train(
        Table(X, y),
        algorithm="dt",
        tree_type=tree_type,
        canaries=0,
        min_samples_leaf=50,
    )
    optimized = model.optimize_inference(physical_cores=1)

    assert np.allclose(
        optimized.predict_proba([[0.0, 0.0, 0.0]]),
        model.predict_proba([[0.0, 0.0, 0.0]]),
        atol=PREDICTION_TOLERANCE,
        rtol=PREDICTION_TOLERANCE,
    )


@pytest.mark.parametrize("tree_type", ["id3", "c45", "cart", "randomized", "oblivious"])
def test_random_forest_optimized_classifier_predict_proba_matches_base_model_for_all_tree_types(
    tree_type: str,
) -> None:
    rng = np.random.default_rng(7)
    X = rng.standard_normal((2_000, 3))
    y = rng.integers(0, 2, size=2_000)
    model = train(
        Table(X, y),
        algorithm="rf",
        tree_type=tree_type,
        n_trees=100,
        min_samples_leaf=50,
        max_features=2,
    )
    optimized = model.optimize_inference(physical_cores=1)

    assert np.allclose(
        optimized.predict_proba([[0.0, 0.0, 0.0]]),
        model.predict_proba([[0.0, 0.0, 0.0]]),
        atol=PREDICTION_TOLERANCE,
        rtol=PREDICTION_TOLERANCE,
    )
    assert np.allclose(
        optimized.predict_proba(X),
        model.predict_proba(X),
        atol=PREDICTION_TOLERANCE,
        rtol=PREDICTION_TOLERANCE,
    )
    assert np.allclose(
        optimized.predict_proba(X),
        model.predict_proba(X),
        atol=PREDICTION_TOLERANCE,
        rtol=PREDICTION_TOLERANCE,
    )


def test_compiled_optimized_model_round_trips() -> None:
    X = np.array(
        [
            [0.0, 10.0, 20.0],
            [1.0, 10.0, 20.0],
            [2.0, 10.0, 20.0],
            [3.0, 10.0, 20.0],
        ]
    )
    y = np.array([10.0, 12.0, 14.0, 20.0])
    model = train(X, y, task="regression", tree_type="cart", canaries=0)
    optimized = model.optimize_inference(physical_cores=1)
    compiled = optimized.serialize_compiled()
    restored = OptimizedModel.deserialize_compiled(compiled, physical_cores=1)

    assert isinstance(compiled, bytes)
    assert optimized.serialize() == restored.serialize()
    assert optimized.to_ir_json() == restored.to_ir_json()
    assert optimized.used_feature_indices == [0]
    assert restored.used_feature_indices == [0]
    assert np.allclose(
        optimized.predict(X),
        restored.predict(X),
        atol=PREDICTION_TOLERANCE,
        rtol=PREDICTION_TOLERANCE,
    )


def test_optimized_random_forest_projects_inputs_to_used_features() -> None:
    X = np.array(
        [
            [0.0, 0.0, 5.0, 9.0],
            [0.0, 1.0, 5.0, 9.0],
            [1.0, 0.0, 5.0, 9.0],
            [1.0, 1.0, 5.0, 9.0],
        ]
    )
    y = np.array([0.0, 0.0, 0.0, 1.0])
    model = train(
        X,
        y,
        algorithm="rf",
        task="classification",
        tree_type="cart",
        n_trees=8,
        max_features=2,
        seed=7,
        canaries=0,
    )
    optimized = model.optimize_inference(physical_cores=1)

    assert model.used_feature_indices == [0, 1]
    assert optimized.used_feature_indices == [0, 1]
    assert optimized.used_feature_count == 2
    assert np.allclose(
        optimized.predict(X),
        model.predict(X),
        atol=PREDICTION_TOLERANCE,
        rtol=PREDICTION_TOLERANCE,
    )


def test_compiled_optimized_model_rejects_invalid_bytes() -> None:
    with pytest.raises(ValueError):
        OptimizedModel.deserialize_compiled(b"not-a-compiled-artifact")


@pytest.mark.parametrize("tree_type", ["id3", "c45", "cart", "randomized"])
def test_standard_tree_introspection_exposes_structure_and_nodes(
    tree_type: str,
) -> None:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = train(X, y, task="classification", tree_type=tree_type, canaries=0)

    structure = model.tree_structure()
    stats = model.tree_prediction_stats()
    root = model.tree_node(0)
    first_leaf = model.tree_leaf(0)

    assert model.tree_count == 1
    assert structure["representation"] == "node_tree"
    assert structure["leaf_count"] >= 2
    assert structure["actual_depth"] >= 1
    assert structure["shortest_path"] <= structure["longest_path"]
    assert stats["count"] >= 2
    assert stats["unique_count"] >= 1
    assert root["kind"] in {"binary_branch", "multiway_branch", "leaf"}
    assert first_leaf["leaf"]["prediction_kind"] == "class_index"


def test_oblivious_tree_introspection_exposes_levels_and_leaves() -> None:
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = train(X, y, task="classification", tree_type="oblivious", canaries=0)

    structure = model.tree_structure()
    level = model.tree_level(0)
    leaf = model.tree_leaf(0)

    assert structure["representation"] == "oblivious_levels"
    assert structure["actual_depth"] >= 1
    assert structure["shortest_path"] == structure["longest_path"]
    assert level["split"]["split_type"] in {"numeric_bin_threshold", "boolean_test"}
    assert leaf["leaf"]["prediction_kind"] == "class_index"


@pytest.mark.parametrize("tree_type", ["cart", "randomized", "oblivious"])
def test_regression_tree_prediction_stats_are_numeric(tree_type: str) -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 4.0, 9.0])
    model = train(X, y, task="regression", tree_type=tree_type, canaries=0)

    stats = model.tree_prediction_stats()

    assert stats["count"] >= 2
    assert np.isfinite(stats["min"])
    assert np.isfinite(stats["max"])
    assert np.isfinite(stats["mean"])
    assert np.isfinite(stats["std_dev"])


def test_optimized_tree_introspection_matches_base_model() -> None:
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = train(X, y, task="classification", tree_type="oblivious", canaries=0)
    optimized = model.optimize_inference(physical_cores=1)

    assert optimized.tree_count == model.tree_count
    assert optimized.tree_structure() == model.tree_structure()
    assert optimized.tree_prediction_stats() == model.tree_prediction_stats()
    assert optimized.tree_level(0) == model.tree_level(0)
    assert optimized.tree_leaf(0) == model.tree_leaf(0)


def test_random_forest_tree_introspection_accepts_tree_index() -> None:
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    forest = train(
        X,
        y,
        algorithm="rf",
        task="classification",
        tree_type="cart",
        canaries=0,
        n_trees=3,
        seed=7,
    )

    structure = forest.tree_structure(tree_index=1)
    node = forest.tree_node(0, tree_index=1)

    assert forest.tree_count == 3
    assert structure["leaf_count"] >= 1
    assert node["kind"] in {"binary_branch", "multiway_branch", "leaf"}


@pytest.mark.parametrize("tree_type", ["cart", "id3"])
def test_model_to_dataframe_exposes_standard_tree_rows(tree_type: str) -> None:
    try:
        import importlib

        frame_lib = importlib.import_module("polars")

        expected_type = frame_lib.DataFrame
    except ModuleNotFoundError:
        pa = pytest.importorskip("pyarrow")
        frame_lib = pa
        expected_type = pa.Table
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = train(X, y, task="classification", tree_type=tree_type, canaries=0)

    df = model.to_dataframe()

    assert isinstance(df, expected_type)
    if expected_type.__name__ == "DataFrame":
        assert df.height > 0
        assert "leaf" in set(df["node_type"].to_list())
        assert set(df["representation"].to_list()) == {"node_tree"}
        assert "split_feature" in df.columns
        assert "leaf_value" in df.columns
    else:
        assert df.num_rows > 0
        assert "leaf" in set(df.column("node_type").to_pylist())
        assert set(df.column("representation").to_pylist()) == {"node_tree"}
        assert "split_feature" in df.column_names
        assert "leaf_value" in df.column_names


def test_model_to_dataframe_exposes_oblivious_tree_rows() -> None:
    try:
        import importlib

        frame_lib = importlib.import_module("polars")

        expected_type = frame_lib.DataFrame
    except ModuleNotFoundError:
        pa = pytest.importorskip("pyarrow")
        frame_lib = pa
        expected_type = pa.Table
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = train(X, y, task="classification", tree_type="oblivious", canaries=0)

    df = model.to_dataframe()

    assert isinstance(df, expected_type)
    if expected_type.__name__ == "DataFrame":
        assert df.height > 0
        assert set(df["representation"].to_list()) == {"oblivious_levels"}
        assert {"level", "leaf"}.issubset(set(df["node_type"].to_list()))
    else:
        assert df.num_rows > 0
        assert set(df.column("representation").to_pylist()) == {"oblivious_levels"}
        assert {"level", "leaf"}.issubset(set(df.column("node_type").to_pylist()))


def test_random_forest_to_dataframe_supports_tree_index_filtering() -> None:
    try:
        import importlib

        frame_lib = importlib.import_module("polars")

        expected_type = frame_lib.DataFrame
    except ModuleNotFoundError:
        pa = pytest.importorskip("pyarrow")
        frame_lib = pa
        expected_type = pa.Table
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    forest = train(
        X,
        y,
        algorithm="rf",
        task="classification",
        tree_type="cart",
        canaries=0,
        n_trees=3,
        seed=7,
    )

    full_df = forest.to_dataframe()
    tree_df = forest.to_dataframe(tree_index=1)

    assert isinstance(full_df, expected_type)
    assert isinstance(tree_df, expected_type)
    if expected_type.__name__ == "DataFrame":
        assert set(full_df["tree_index"].to_list()) == {0, 1, 2}
        assert set(tree_df["tree_index"].to_list()) == {1}
    else:
        assert set(full_df.column("tree_index").to_pylist()) == {0, 1, 2}
        assert set(tree_df.column("tree_index").to_pylist()) == {1}


def test_optimized_model_to_dataframe_matches_base_model() -> None:
    pa: object
    try:
        import importlib

        pl = importlib.import_module("polars")
    except ModuleNotFoundError:
        pl = None
    if pl is None:
        pa = pytest.importorskip("pyarrow")
    X = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = train(X, y, task="classification", tree_type="oblivious", canaries=0)
    optimized = model.optimize_inference(physical_cores=1)

    base_df = model.to_dataframe()
    optimized_df = optimized.to_dataframe()

    if pl is not None:
        base_df = base_df.sort(["tree_index", "node_index"])
        optimized_df = optimized_df.sort(["tree_index", "node_index"])
        assert isinstance(base_df, pl.DataFrame)
        assert base_df.equals(optimized_df)
    else:
        assert hasattr(pa, "Table")
        assert isinstance(base_df, pa.Table)
        assert base_df.equals(optimized_df)


def test_train_id3_classifier(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(
        X, y, algorithm="dt", task="classification", tree_type="id3", canaries=0
    )

    assert model.algorithm == "dt"
    assert model.task == "classification"
    assert model.criterion == "entropy"
    assert model.tree_type == "id3"
    assert np.array_equal(model.predict(X), y)


def test_train_c45_classifier(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(
        X, y, algorithm="dt", task="classification", tree_type="c45", canaries=0
    )

    assert model.algorithm == "dt"
    assert model.task == "classification"
    assert model.criterion == "entropy"
    assert model.tree_type == "c45"
    assert np.array_equal(model.predict(X), y)


def test_train_oblivious_classifier(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(
        X, y, algorithm="dt", task="classification", tree_type="oblivious", canaries=0
    )

    assert model.algorithm == "dt"
    assert model.task == "classification"
    assert model.criterion == "gini"
    assert model.tree_type == "oblivious"
    assert np.array_equal(model.predict(X), y)


def test_train_randomized_classifier(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(
        X, y, algorithm="dt", task="classification", tree_type="randomized", canaries=0
    )

    assert model.algorithm == "dt"
    assert model.task == "classification"
    assert model.criterion == "gini"
    assert model.tree_type == "randomized"
    assert np.array_equal(model.predict(X), y)


def test_train_cart_regressor() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0])

    model = train(
        X,
        y,
        algorithm="dt",
        task="regression",
        tree_type="cart",
        canaries=0,
        bins=64,
    )

    assert model.algorithm == "dt"
    assert model.task == "regression"
    assert model.criterion == "mean"
    assert model.tree_type == "cart"
    assert np.array_equal(model.predict(X), y)


def test_train_oblivious_regressor() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0])

    model = train(
        X,
        y,
        algorithm="dt",
        task="regression",
        tree_type="oblivious",
        canaries=0,
        bins=64,
    )

    assert model.algorithm == "dt"
    assert model.task == "regression"
    assert model.criterion == "mean"
    assert model.tree_type == "oblivious"
    assert np.array_equal(model.predict(X), y)


def test_train_randomized_regressor() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0])

    model = train(
        X,
        y,
        algorithm="dt",
        task="regression",
        tree_type="randomized",
        canaries=0,
        bins=64,
    )

    assert model.algorithm == "dt"
    assert model.task == "regression"
    assert model.criterion == "mean"
    assert model.tree_type == "randomized"
    preds = model.predict(X)
    baseline = np.full_like(y, fill_value=np.mean(y))
    assert np.sum((preds - y) ** 2) < np.sum((baseline - y) ** 2)


@pytest.mark.parametrize(
    ("task", "tree_type", "expected_criterion"),
    [
        ("regression", "cart", "mean"),
        ("regression", "randomized", "mean"),
        ("regression", "oblivious", "mean"),
        ("classification", "id3", "entropy"),
        ("classification", "c45", "entropy"),
        ("classification", "cart", "gini"),
        ("classification", "randomized", "gini"),
        ("classification", "oblivious", "gini"),
    ],
)
def test_auto_criterion_resolves_by_task_and_tree_type(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
    toy_data: tuple[NDArray[np.float64], NDArray[np.float64]],
    task: str,
    tree_type: str,
    expected_criterion: str,
) -> None:
    if task == "classification":
        X, y = and_data
    else:
        X, y = toy_data

    model = train(X, y, task=task, tree_type=tree_type, criterion="auto", canaries=0)

    assert model.criterion == expected_criterion


def test_train_cart_regressor_can_use_median_criterion() -> None:
    X = np.zeros((3, 1))
    y = np.array([0.0, 0.0, 100.0])

    mean_model = train(
        X, y, task="regression", tree_type="cart", criterion="mean", canaries=0
    )
    median_model = train(
        X, y, task="regression", tree_type="cart", criterion="median", canaries=0
    )

    assert mean_model.criterion == "mean"
    assert median_model.criterion == "median"
    assert np.allclose(mean_model.predict(X), np.array([100.0 / 3.0] * 3))
    assert np.array_equal(median_model.predict(X), np.array([0.0, 0.0, 0.0]))


def test_train_oblivious_regressor_can_use_median_criterion() -> None:
    X = np.zeros((3, 1))
    y = np.array([0.0, 0.0, 100.0])

    mean_model = train(
        X, y, task="regression", tree_type="oblivious", criterion="mean", canaries=0
    )
    median_model = train(
        X, y, task="regression", tree_type="oblivious", criterion="median", canaries=0
    )

    assert mean_model.criterion == "mean"
    assert median_model.criterion == "median"
    assert np.allclose(mean_model.predict(X), np.array([100.0 / 3.0] * 3))
    assert np.array_equal(median_model.predict(X), np.array([0.0, 0.0, 0.0]))


def test_train_rejects_unknown_algorithm() -> None:
    X = np.zeros((2, 1))
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="Unsupported algorithm"):
        train(X, y, algorithm="unknown")


def test_train_gradient_boosting_regressor_learns_signal() -> None:
    X = np.array([[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]], dtype=np.float64)
    y = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float64)

    model = train(
        X,
        y,
        algorithm="gbm",
        task="regression",
        tree_type="cart",
        n_trees=25,
        learning_rate=0.2,
        top_gradient_fraction=1.0,
        other_gradient_fraction=0.0,
        canaries=0,
    )

    preds = model.predict(X)
    assert model.algorithm == "gbm"
    assert model.criterion == "second_order"
    assert model.learning_rate == 0.2
    assert model.bootstrap is False
    assert np.unique(preds).size > 1
    assert preds[0] < preds[-1]


def test_train_gradient_boosting_classifier_predicts_probabilities() -> None:
    X = np.array([[0.0], [0.1], [0.9], [1.0]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.int64)

    model = train(
        X,
        y,
        algorithm="gbm",
        task="classification",
        tree_type="cart",
        n_trees=25,
        learning_rate=0.2,
        bootstrap=True,
        top_gradient_fraction=0.5,
        other_gradient_fraction=0.25,
        canaries=0,
    )

    probs = model.predict_proba(X)
    assert model.algorithm == "gbm"
    assert model.learning_rate == 0.2
    assert model.bootstrap is True
    assert model.top_gradient_fraction == 0.5
    assert model.other_gradient_fraction == 0.25
    assert probs.shape == (4, 2)
    assert probs[0, 1] < 0.5
    assert probs[-1, 1] > 0.5


def test_train_gradient_boosting_rejects_multiclass_classification() -> None:
    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([0, 1, 2], dtype=np.int64)

    with pytest.raises(ValueError, match="binary classification only"):
        train(
            X, y, algorithm="gbm", task="classification", tree_type="cart", canaries=0
        )


def test_train_accepts_fixed_bins() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 4.0, 9.0])

    model = train(X, y, task="regression", tree_type="cart", bins=64, canaries=0)

    preds = model.predict(X)
    assert preds.shape == (4,)


def test_train_accepts_histogram_bins() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 4.0, 9.0])

    model = train(
        X,
        y,
        task="regression",
        tree_type="cart",
        bins=64,
        histogram_bins=8,
        canaries=0,
    )

    preds = model.predict(X)
    assert preds.shape == (4,)


def test_train_accepts_histogram_bins_for_prebuilt_table(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    table = Table(X, y, canaries=0, bins=64)

    model = train(
        table,
        task="classification",
        tree_type="cart",
        histogram_bins=8,
    )

    assert np.array_equal(model.predict(table), y)


def test_train_rejects_invalid_bins() -> None:
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="between 1 and 128"):
        train(X, y, bins=256)


def test_train_rejects_invalid_histogram_bins() -> None:
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="between 1 and 128"):
        train(X, y, histogram_bins=256)


def test_train_rejects_y_when_x_is_already_a_table(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    table = Table(X, y, canaries=0)

    with pytest.raises(ValueError, match="y must be omitted"):
        train(table, y)


def test_train_rejects_bins_when_x_is_already_a_table(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    table = Table(X, y, canaries=0)

    with pytest.raises(
        ValueError, match="bins must be omitted when x is already a Table"
    ):
        train(table, task="classification", tree_type="cart", bins=64)


def test_train_rejects_unknown_task() -> None:
    X = np.zeros((2, 1))
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="Unsupported task"):
        train(X, y, task="ranking")


def test_train_rejects_unknown_criterion() -> None:
    X = np.zeros((2, 1))
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="Unsupported criterion"):
        train(X, y, criterion="mae")


@pytest.mark.parametrize(
    ("task", "tree_type"),
    [("regression", "cart"), ("classification", "cart")],
)
def test_train_rejects_non_finite_targets(task: str, tree_type: str) -> None:
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, np.nan])

    with pytest.raises(ValueError, match="must be finite"):
        train(X, y, task=task, tree_type=tree_type, canaries=0)


@pytest.mark.parametrize(
    ("task", "tree_type", "criterion"),
    [
        ("regression", "id3", "auto"),
        ("regression", "c45", "auto"),
        ("classification", "cart", "mean"),
    ],
)
def test_train_rejects_unsupported_task_tree_type_pairs(
    task: str,
    tree_type: str,
    criterion: str,
) -> None:
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="Unsupported training configuration"):
        train(X, y, task=task, tree_type=tree_type, criterion=criterion)


def test_train_accepts_canaries_hyperparameter() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    model = train(
        X, y, algorithm="dt", task="classification", tree_type="cart", canaries=1
    )

    assert model.algorithm == "dt"


def test_train_accepts_physical_cores_parameter() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    model = train(
        X,
        y,
        algorithm="dt",
        task="classification",
        tree_type="cart",
        canaries=0,
        physical_cores=1,
    )

    assert model.algorithm == "dt"


def test_train_caps_large_physical_core_requests() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    single_core_model = train(
        X,
        y,
        algorithm="dt",
        task="classification",
        tree_type="cart",
        canaries=0,
        physical_cores=1,
    )
    overprovisioned_model = train(
        X,
        y,
        algorithm="dt",
        task="classification",
        tree_type="cart",
        canaries=0,
        physical_cores=10_000,
    )

    assert np.array_equal(
        single_core_model.predict(X), overprovisioned_model.predict(X)
    )


def test_train_rejects_zero_physical_cores() -> None:
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="Requested 0 physical cores"):
        train(X, y, physical_cores=0)


def test_predict_generalizes_on_binary_feature_rows(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(
        X, y, algorithm="dt", task="classification", tree_type="cart", canaries=0
    )

    new_rows = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])

    assert np.array_equal(model.predict(new_rows), np.array([0.0, 1.0, 0.0]))


def test_table_accepts_plain_python_lists() -> None:
    X = [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [0.0, 1.0, 1.0]

    table = Table(X, y, canaries=0)

    assert table.kind == "sparse"
    model = train(table, task="classification", tree_type="cart")
    assert np.array_equal(model.predict(X), np.array(y))


def test_table_accepts_pandas_dataframes_if_installed() -> None:
    pd = pytest.importorskip("pandas")

    X = pd.DataFrame({"a": [0.0, 1.0, 1.0], "b": [1.0, 0.0, 1.0]})
    y = pd.Series([0.0, 1.0, 1.0])

    table = Table(X, y, canaries=0)

    assert table.kind == "sparse"
    model = train(table, task="classification", tree_type="cart")
    assert np.array_equal(model.predict(X), y.to_numpy())


def test_table_accepts_polars_dataframes_if_installed() -> None:
    pl = pytest.importorskip("polars")

    X = pl.DataFrame({"a": [0.0, 1.0, 1.0], "b": [1.0, 0.0, 1.0]})
    y = pl.Series("y", [0.0, 1.0, 1.0])

    table = Table(X, y, canaries=0)

    assert table.kind == "sparse"
    model = train(table, task="classification", tree_type="cart")
    assert np.array_equal(model.predict(X), np.array([0.0, 1.0, 1.0]))


def test_predict_accepts_polars_lazyframes_if_installed() -> None:
    pl = pytest.importorskip("polars")

    X = pl.DataFrame({"a": [0.0, 1.0, 1.0], "b": [1.0, 0.0, 1.0]})
    y = np.array([0.0, 1.0, 1.0])

    model = train(X, y, task="classification", tree_type="cart", canaries=0)
    preds = model.predict(X.lazy())

    assert np.array_equal(preds, y)


def test_predict_batches_large_polars_lazyframes_if_installed() -> None:
    pl = pytest.importorskip("polars")

    pattern_a = np.array([0.0, 1.0, 1.0])
    pattern_b = np.array([1.0, 0.0, 1.0])
    pattern_y = np.array([0.0, 1.0, 1.0])
    n_rows = 20_003
    X = pl.DataFrame(
        {
            "a": np.resize(pattern_a, n_rows),
            "b": np.resize(pattern_b, n_rows),
        }
    )
    expected = np.resize(pattern_y, n_rows)

    model = train(
        pl.DataFrame({"a": pattern_a, "b": pattern_b}),
        pattern_y,
        task="classification",
        tree_type="cart",
        canaries=0,
    )
    optimized = model.optimize_inference(physical_cores=1)
    optimized_dataframe_preds = optimized.predict(X)
    optimized_lazyframe_preds = optimized.predict(X.lazy())

    assert np.allclose(
        model.predict(X.lazy()),
        expected,
        atol=PREDICTION_TOLERANCE,
        rtol=PREDICTION_TOLERANCE,
    )
    assert np.allclose(
        optimized_lazyframe_preds,
        expected,
        atol=PREDICTION_TOLERANCE,
        rtol=PREDICTION_TOLERANCE,
    )
    assert np.allclose(
        optimized_lazyframe_preds,
        optimized_dataframe_preds,
        atol=PREDICTION_TOLERANCE,
        rtol=PREDICTION_TOLERANCE,
    )


def test_table_accepts_pyarrow_tables_if_installed() -> None:
    pa = pytest.importorskip("pyarrow")

    X = pa.table({"a": [0.0, 1.0, 1.0], "b": [1.0, 0.0, 1.0]})
    y = pa.array([0.0, 1.0, 1.0])

    table = Table(X, y, canaries=0)

    assert table.kind == "sparse"
    model = train(table, task="classification", tree_type="cart")
    assert np.array_equal(model.predict(X), np.array([0.0, 1.0, 1.0]))


def test_table_accepts_scipy_sparse_matrices_if_installed() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")

    X = scipy_sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0])

    table = Table(X, y, canaries=0)

    assert table.kind == "sparse"
    model = train(table, task="classification", tree_type="cart")
    assert np.array_equal(model.predict(X), y)


def test_predict_accepts_scipy_sparse_matrices_if_installed() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")

    X = scipy_sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0])

    model = train(X, y, task="classification", tree_type="cart", canaries=0)

    assert np.array_equal(model.predict(X), y)


def test_table_rejects_non_binary_scipy_sparse_matrices_if_installed() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")

    X = scipy_sparse.csr_matrix([[0.0, 2.0], [1.0, 0.0]])
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="binary sparse inputs"):
        Table(X, y, canaries=0)


def test_train_accepts_scipy_dense_matrix_like_inputs_if_installed() -> None:
    scipy_sparse = pytest.importorskip("scipy.sparse")

    X = scipy_sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).todense()
    y = scipy_sparse.csr_matrix([[0.0], [1.0], [1.0]]).todense()

    model = train(X, y, task="classification", tree_type="cart", canaries=0)

    assert model.task == "classification"
    assert np.array_equal(model.predict(X), np.array([0.0, 1.0, 1.0]))


def test_model_to_ir_json_exports_regressor_metadata() -> None:
    X = np.array([[0.0, 0.0], [1.0, 10.0], [0.0, 20.0], [1.0, 30.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])

    model = train(X, y, canaries=2)
    ir = json.loads(model.to_ir_json())

    assert ir["ir_version"] == "1.0.0"
    assert ir["model"]["algorithm"] == "dt"
    assert ir["model"]["tree_type"] == "cart"
    assert ir["training_metadata"]["canaries"] == 2
    assert ir["input_schema"]["feature_count"] == 2


def test_model_to_ir_json_exports_oblivious_tree_structure() -> None:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0, 2.0])

    model = train(X, y, task="regression", tree_type="oblivious", canaries=0)
    ir = json.loads(model.to_ir_json(pretty=True))

    assert ir["model"]["representation"] == "oblivious_levels"
    assert ir["model"]["trees"][0]["representation"] == "oblivious_levels"
    assert ir["model"]["trees"][0]["leaf_indexing"]["bit_order"] == "msb_first"
    assert len(ir["model"]["trees"][0]["leaves"]) == 4


def test_model_serialize_alias_matches_ir_json() -> None:
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, 2.0, 3.0])

    model = train(X, y)

    assert json.loads(model.serialize()) == json.loads(model.to_ir_json())


def test_model_deserialize_round_trip() -> None:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 0.0, 0.0, 1.0])

    model = train(X, y, task="classification", tree_type="cart", canaries=1)
    restored = type(model).deserialize(model.serialize())

    assert restored.algorithm == model.algorithm
    assert restored.task == model.task
    assert restored.tree_type == model.tree_type
    assert restored.criterion == model.criterion
    assert np.array_equal(restored.predict(X), model.predict(X))
