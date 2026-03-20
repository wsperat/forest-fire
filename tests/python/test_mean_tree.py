import json

import numpy as np
import pytest
from forestfire import Table, train
from numpy.typing import NDArray


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


def test_train_and_predict_shape_and_value(
    toy_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = toy_data
    m = train(X, y, algorithm="dt", tree_type="target_mean")
    assert m.algorithm == "dt"
    assert m.task == "regression"
    assert m.criterion == "mean"
    assert m.tree_type == "target_mean"
    assert m.mean_ == pytest.approx(14.0, abs=1e-12)
    assert m.mean_ is not None

    preds = m.predict(X)
    assert preds.shape == (X.shape[0],)
    assert np.allclose(preds, m.mean_)


def test_train_defaults_match_documented_regression_baseline(
    toy_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = toy_data

    model = train(X, y)

    assert model.algorithm == "dt"
    assert model.task == "regression"
    assert model.tree_type == "target_mean"
    assert model.criterion == "mean"


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


def test_train_accepts_prebuilt_table(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    table = Table(X, y, canaries=0)

    model = train(table, task="classification", tree_type="cart")

    assert model.algorithm == "dt"
    assert np.array_equal(model.predict(table), y)


@pytest.mark.parametrize(
    "y, expected",
    [
        (np.array([1.0, 1.0, 1.0]), 1.0),
        (np.array([0.0, 2.0]), 1.0),
        (np.array([5.0]), 5.0),
    ],
)
def test_mean_is_computed_correctly(y: NDArray[np.float64], expected: float) -> None:
    X = np.zeros((y.shape[0], 2))
    m = train(X, y, algorithm="dt", tree_type="target_mean")
    assert m.mean_ == pytest.approx(expected, abs=1e-12)
    assert np.allclose(m.predict(X), expected)


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
    assert model.mean_ is None
    assert np.array_equal(model.predict(X), y)


def test_predict_accepts_feature_only_table(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    model = train(X, y, task="classification", tree_type="cart", canaries=0)

    feature_table = Table(X)

    assert feature_table.kind == "sparse"
    assert np.array_equal(model.predict(feature_table), y)


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
    assert model.mean_ is None
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
    assert model.mean_ is None
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
    assert model.mean_ is None
    assert np.array_equal(model.predict(X), y)


def test_train_cart_regressor() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0])

    model = train(X, y, algorithm="dt", task="regression", tree_type="cart", canaries=0)

    assert model.algorithm == "dt"
    assert model.task == "regression"
    assert model.criterion == "mean"
    assert model.tree_type == "cart"
    assert model.mean_ is None
    assert np.array_equal(model.predict(X), y)


def test_train_oblivious_regressor() -> None:
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


def test_train_target_mean_can_use_median_criterion() -> None:
    X = np.zeros((3, 1))
    y = np.array([0.0, 0.0, 100.0])

    model = train(X, y, criterion="median")

    assert model.task == "regression"
    assert model.criterion == "median"
    assert model.tree_type == "target_mean"
    assert model.mean_ == 0.0
    assert np.array_equal(model.predict(X), np.array([0.0, 0.0, 0.0]))


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
        train(X, y, algorithm="rf")


def test_train_rejects_y_when_x_is_already_a_table(
    and_data: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    X, y = and_data
    table = Table(X, y, canaries=0)

    with pytest.raises(ValueError, match="y must be omitted"):
        train(table, y)


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
        ("classification", "target_mean", "auto"),
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


def test_model_to_ir_json_exports_target_mean_metadata() -> None:
    X = np.array([[0.0, 0.0], [1.0, 10.0], [0.0, 20.0], [1.0, 30.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])

    model = train(X, y, canaries=2)
    ir = json.loads(model.to_ir_json())

    assert ir["ir_version"] == "1.0.0"
    assert ir["model"]["algorithm"] == "dt"
    assert ir["model"]["tree_type"] == "target_mean"
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
