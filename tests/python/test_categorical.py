import numpy as np
from forestfire import train
from forestfire.gbm import ExtraGBMClassifier


def categorical_rows() -> list[list[object]]:
    return [
        ["red", 0.0],
        ["red", 1.0],
        ["blue", 0.0],
        ["blue", 1.0],
        ["green", 0.0],
        ["green", 1.0],
    ]


def test_train_supports_one_hot_categorical_strategy() -> None:
    x = categorical_rows()
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])

    model = train(
        x,
        y,
        task="classification",
        tree_type="cart",
        canaries=0,
        categorical_strategy="one_hot",
    )

    preds = np.asarray(model.predict(x), dtype=np.float64)
    assert preds.shape == y.shape
    assert np.isfinite(preds).all()


def test_train_supports_target_encoded_categorical_strategy() -> None:
    x = categorical_rows()
    y = np.array([0.0, 1.0, 4.0, 5.0, 8.0, 9.0])

    model = train(
        x,
        y,
        task="regression",
        tree_type="cart",
        canaries=0,
        categorical_strategy="target",
    )

    preds = np.asarray(model.predict(x), dtype=np.float64)
    assert preds.shape == y.shape
    assert np.isfinite(preds).all()


def test_train_supports_fisher_categorical_strategy() -> None:
    x = categorical_rows()
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    model = train(
        x,
        y,
        task="classification",
        tree_type="randomized",
        canaries=0,
        categorical_strategy="fisher",
    )

    preds = np.asarray(model.predict(x), dtype=np.float64)
    assert preds.shape == y.shape
    assert np.isfinite(preds).all()


def test_categorical_model_reuses_encoding_for_unseen_prediction_rows() -> None:
    x = categorical_rows()
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    model = train(
        x,
        y,
        task="classification",
        tree_type="cart",
        canaries=0,
        categorical_strategy="target",
    )

    preds = np.asarray(model.predict([["purple", 0.0], ["red", 0.0]]), dtype=np.float64)
    assert preds.shape == (2,)
    assert np.isfinite(preds).all()


def test_gbm_wrapper_accepts_categorical_strategy() -> None:
    x = categorical_rows()
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    model = ExtraGBMClassifier(
        n_estimators=5,
        categorical_strategy="one_hot",
    ).fit(x, y)

    proba = model.predict_proba([["red", 0.0], ["blue", 1.0]])
    assert proba.shape == (2, 2)
    assert np.isfinite(proba).all()
