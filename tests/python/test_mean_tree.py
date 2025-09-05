import numpy as np
import pytest
from forestfire import TargetMeanTree


@pytest.fixture
def toy_data():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([10.0, 12.0, 14.0, 20.0])
    return X, y


def test_fit_and_predict_shape_and_value(toy_data):
    X, y = toy_data
    m = TargetMeanTree.fit(X, y)
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
    m = TargetMeanTree.fit(X, y)
    assert m.mean_ == pytest.approx(expected, abs=1e-12)
    assert np.allclose(m.predict(X), expected)


def test_fit_raises_on_mismatched_lengths():
    X = np.zeros((3, 1))
    y = np.ones(2)
    with pytest.raises(ValueError):
        TargetMeanTree.fit(X, y)
