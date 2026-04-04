# mypy: disable-error-code="import-not-found,import-untyped"

import numpy as np
import pytest
from forestfire.forest import CARTRandomForestRegressor, ExtraRandomForestClassifier
from forestfire.gbm import CARTGBMRegressor, ExtraGBMClassifier, ObliviousGBMRegressor
from forestfire.tree import CARTClassifier, ExtraRegressor, ObliviousRegressor

sklearn = pytest.importorskip("sklearn")
clone = sklearn.base.clone


def test_tree_wrapper_uses_expected_backend_defaults() -> None:
    x = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    model = CARTClassifier().fit(x, y)

    assert model.model_.algorithm == "dt"
    assert model.model_.tree_type == "cart"
    assert model.model_.criterion == "gini"
    assert np.array_equal(model.predict(x), y)
    assert np.array_equal(model.classes_, np.array([0.0, 1.0]))
    assert model.n_features_in_ == 1


def test_tree_regressor_wrapper_exposes_requested_import_path() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 4.0, 9.0])

    model = ObliviousRegressor(max_depth=3).fit(x, y)

    assert model.model_.algorithm == "dt"
    assert model.model_.tree_type == "oblivious"
    assert model.model_.criterion == "mean"
    assert model.predict(x).shape == (4,)


def test_random_forest_wrapper_uses_expected_backend_defaults() -> None:
    x = np.array([[0.0], [0.0], [1.0], [1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    model = ExtraRandomForestClassifier(n_estimators=7, random_state=3).fit(x, y)

    assert model.model_.algorithm == "rf"
    assert model.model_.tree_type == "randomized"
    assert model.model_.n_trees == 7
    assert model.model_.seed == 3
    assert model.predict_proba(x).shape == (4, 2)


def test_random_forest_regressor_wrapper_is_cloneable() -> None:
    estimator = CARTRandomForestRegressor(
        n_estimators=11,
        max_depth=5,
        max_features="sqrt",
        random_state=9,
    )

    cloned = clone(estimator)

    assert isinstance(cloned, CARTRandomForestRegressor)
    assert cloned.get_params() == estimator.get_params()


def test_gbm_wrappers_use_expected_backend_defaults() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_reg = np.array([0.0, 1.0, 4.0, 9.0])
    y_clf = np.array([0.0, 0.0, 1.0, 1.0])

    regressor = ObliviousGBMRegressor(n_estimators=9).fit(x, y_reg)
    classifier = ExtraGBMClassifier(n_estimators=5, random_state=4).fit(x, y_clf)

    assert regressor.model_.algorithm == "gbm"
    assert regressor.model_.tree_type == "oblivious"
    assert regressor.model_.n_trees == 9
    assert classifier.model_.algorithm == "gbm"
    assert classifier.model_.tree_type == "randomized"
    assert classifier.model_.n_trees == 5
    assert classifier.predict_proba(x).shape == (4, 2)


def test_wrapper_modules_cover_requested_names() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 4.0, 9.0])

    assert ExtraRegressor().fit(x, y).model_.tree_type == "randomized"
    assert CARTRandomForestRegressor().fit(x, y).model_.tree_type == "cart"
    assert CARTGBMRegressor().fit(x, y).model_.tree_type == "cart"
