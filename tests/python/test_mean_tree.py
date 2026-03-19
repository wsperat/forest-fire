import numpy as np
import pytest
from forestfire import train


@pytest.fixture
def toy_data():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([10.0, 12.0, 14.0, 20.0])
    return X, y


@pytest.fixture
def and_data():
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


def test_train_and_predict_shape_and_value(toy_data):
    X, y = toy_data
    m = train(X, y, algorithm="dt", tree_type="target_mean")
    assert m.algorithm == "dt"
    assert m.task == "regression"
    assert m.criterion == "mean"
    assert m.tree_type == "target_mean"
    assert m.mean_ == pytest.approx(14.0, abs=1e-12)

    preds = m.predict(X)
    assert preds.shape == (X.shape[0],)
    assert np.allclose(preds, m.mean_)


def test_train_defaults_match_documented_regression_baseline(toy_data):
    X, y = toy_data

    model = train(X, y)

    assert model.algorithm == "dt"
    assert model.task == "regression"
    assert model.tree_type == "target_mean"
    assert model.criterion == "mean"


@pytest.mark.parametrize(
    "y, expected",
    [
        (np.array([1.0, 1.0, 1.0]), 1.0),
        (np.array([0.0, 2.0]), 1.0),
        (np.array([5.0]), 5.0),
    ],
)
def test_mean_is_computed_correctly(y, expected):
    X = np.zeros((y.shape[0], 2))
    m = train(X, y, algorithm="dt", tree_type="target_mean")
    assert m.mean_ == pytest.approx(expected, abs=1e-12)
    assert np.allclose(m.predict(X), expected)


def test_train_raises_on_mismatched_lengths():
    X = np.zeros((3, 1))
    y = np.ones(2)
    with pytest.raises(ValueError):
        train(X, y)


def test_train_cart_classifier(and_data):
    X, y = and_data
    model = train(
        X, y, algorithm="dt", task="classification", tree_type="cart", canaries=0
    )

    assert model.algorithm == "dt"
    assert model.task == "classification"
    assert model.criterion == "gini"
    assert model.tree_type == "cart"
    assert model.mean_ is None
    assert np.array_equal(model.predict(X), y)


def test_train_id3_classifier(and_data):
    X, y = and_data
    model = train(
        X, y, algorithm="dt", task="classification", tree_type="id3", canaries=0
    )

    assert model.algorithm == "dt"
    assert model.task == "classification"
    assert model.criterion == "entropy"
    assert model.tree_type == "id3"
    assert model.mean_ is None
    assert np.array_equal(model.predict(X), y)


def test_train_c45_classifier(and_data):
    X, y = and_data
    model = train(
        X, y, algorithm="dt", task="classification", tree_type="c45", canaries=0
    )

    assert model.algorithm == "dt"
    assert model.task == "classification"
    assert model.criterion == "entropy"
    assert model.tree_type == "c45"
    assert model.mean_ is None
    assert np.array_equal(model.predict(X), y)


def test_train_oblivious_classifier(and_data):
    X, y = and_data
    model = train(
        X, y, algorithm="dt", task="classification", tree_type="oblivious", canaries=0
    )

    assert model.algorithm == "dt"
    assert model.task == "classification"
    assert model.criterion == "gini"
    assert model.tree_type == "oblivious"
    assert model.mean_ is None
    assert np.array_equal(model.predict(X), y)


def test_train_cart_regressor():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0])

    model = train(X, y, algorithm="dt", task="regression", tree_type="cart", canaries=0)

    assert model.algorithm == "dt"
    assert model.task == "regression"
    assert model.criterion == "mean"
    assert model.tree_type == "cart"
    assert model.mean_ is None
    assert np.array_equal(model.predict(X), y)


def test_train_oblivious_regressor():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0])

    model = train(
        X, y, algorithm="dt", task="regression", tree_type="oblivious", canaries=0
    )

    assert model.algorithm == "dt"
    assert model.task == "regression"
    assert model.criterion == "mean"
    assert model.tree_type == "oblivious"
    assert model.mean_ is None
    assert np.array_equal(model.predict(X), y)


@pytest.mark.parametrize(
    ("task", "tree_type", "expected_criterion"),
    [
        ("regression", "target_mean", "mean"),
        ("regression", "cart", "mean"),
        ("regression", "oblivious", "mean"),
        ("classification", "id3", "entropy"),
        ("classification", "c45", "entropy"),
        ("classification", "cart", "gini"),
        ("classification", "oblivious", "gini"),
    ],
)
def test_auto_criterion_resolves_by_task_and_tree_type(
    and_data, toy_data, task, tree_type, expected_criterion
):
    if task == "classification":
        X, y = and_data
    else:
        X, y = toy_data

    model = train(X, y, task=task, tree_type=tree_type, criterion="auto", canaries=0)

    assert model.criterion == expected_criterion


def test_train_target_mean_can_use_median_criterion():
    X = np.zeros((3, 1))
    y = np.array([0.0, 0.0, 100.0])

    model = train(X, y, criterion="median")

    assert model.task == "regression"
    assert model.criterion == "median"
    assert model.tree_type == "target_mean"
    assert model.mean_ == 0.0
    assert np.array_equal(model.predict(X), np.array([0.0, 0.0, 0.0]))


def test_train_cart_regressor_can_use_median_criterion():
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


def test_train_oblivious_regressor_can_use_median_criterion():
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


def test_train_rejects_unknown_algorithm():
    X = np.zeros((2, 1))
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="Unsupported algorithm"):
        train(X, y, algorithm="rf")


def test_train_rejects_unknown_task():
    X = np.zeros((2, 1))
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="Unsupported task"):
        train(X, y, task="ranking")


def test_train_rejects_unknown_criterion():
    X = np.zeros((2, 1))
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="Unsupported criterion"):
        train(X, y, criterion="mae")


@pytest.mark.parametrize(
    ("task", "tree_type"),
    [("regression", "cart"), ("classification", "cart")],
)
def test_train_rejects_non_finite_targets(task, tree_type):
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, np.nan])

    with pytest.raises(ValueError, match="must be finite"):
        train(X, y, task=task, tree_type=tree_type, canaries=0)


@pytest.mark.parametrize(
    ("task", "tree_type", "criterion"),
    [
        ("regression", "id3", "auto"),
        ("regression", "c45", "auto"),
        ("classification", "target_mean", "auto"),
        ("classification", "cart", "mean"),
    ],
)
def test_train_rejects_unsupported_task_tree_type_pairs(task, tree_type, criterion):
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="Unsupported training configuration"):
        train(X, y, task=task, tree_type=tree_type, criterion=criterion)


def test_train_accepts_canaries_hyperparameter():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    model = train(
        X, y, algorithm="dt", task="classification", tree_type="cart", canaries=1
    )

    assert model.algorithm == "dt"


def test_train_accepts_physical_cores_parameter():
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


def test_train_caps_large_physical_core_requests():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    model = train(
        X,
        y,
        algorithm="dt",
        task="classification",
        tree_type="cart",
        canaries=0,
        physical_cores=10_000,
    )

    assert np.array_equal(model.predict(X), y)


def test_train_rejects_zero_physical_cores():
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="Requested 0 physical cores"):
        train(X, y, physical_cores=0)


def test_predict_generalizes_on_binary_feature_rows(and_data):
    X, y = and_data
    model = train(
        X, y, algorithm="dt", task="classification", tree_type="cart", canaries=0
    )

    new_rows = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])

    assert np.array_equal(model.predict(new_rows), np.array([0.0, 1.0, 0.0]))
