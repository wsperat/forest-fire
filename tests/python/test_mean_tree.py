import numpy as np
import pytest
from forestfire import train


@pytest.fixture
def toy_data():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([10.0, 12.0, 14.0, 20.0])
    return X, y


def test_train_and_predict_shape_and_value(toy_data):
    X, y = toy_data
    m = train(X, y, algorithm="dt", tree_type="target_mean")
    assert m.algorithm == "dt"
    assert m.tree_type == "target_mean"
    assert m.mean_ == pytest.approx(14.0, abs=1e-12)

    preds = m.predict(X)
    assert preds.shape == (X.shape[0],)
    assert np.allclose(preds, m.mean_)


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


def test_train_cart_classifier():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 0.0, 0.0, 1.0])

    model = train(X, y, algorithm="dt", tree_type="cart", canaries=0)

    assert model.algorithm == "dt"
    assert model.tree_type == "cart"
    assert model.mean_ is None
    assert np.array_equal(model.predict(X), y)


def test_train_id3_classifier():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 0.0, 0.0, 1.0])

    model = train(X, y, algorithm="dt", tree_type="id3", canaries=0)

    assert model.algorithm == "dt"
    assert model.tree_type == "id3"
    assert model.mean_ is None
    assert np.array_equal(model.predict(X), y)


def test_train_c45_classifier():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 0.0, 0.0, 1.0])

    model = train(X, y, algorithm="dt", tree_type="c45", canaries=0)

    assert model.algorithm == "dt"
    assert model.tree_type == "c45"
    assert model.mean_ is None
    assert np.array_equal(model.predict(X), y)


def test_train_oblivious_classifier():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 0.0, 0.0, 1.0])

    model = train(X, y, algorithm="dt", tree_type="oblivious", canaries=0)

    assert model.algorithm == "dt"
    assert model.tree_type == "oblivious"
    assert model.mean_ is None
    assert np.array_equal(model.predict(X), y)


def test_train_rejects_unknown_algorithm():
    X = np.zeros((2, 1))
    y = np.array([0.0, 1.0])

    with pytest.raises(ValueError, match="Unsupported algorithm"):
        train(X, y, algorithm="rf")


def test_train_accepts_canaries_hyperparameter():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    model = train(X, y, algorithm="dt", tree_type="cart", canaries=1)

    assert model.algorithm == "dt"
