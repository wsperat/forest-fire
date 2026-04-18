from ._sklearn import _TreeClassifierBase, _TreeRegressorBase


class ID3Classifier(_TreeClassifierBase):
    _tree_type = "id3"


class C45Classifier(_TreeClassifierBase):
    _tree_type = "c45"


class CARTClassifier(_TreeClassifierBase):
    _tree_type = "cart"


class ExtraTreeClassifier(_TreeClassifierBase):
    _tree_type = "randomized"


class ObliviousTreeClassifier(_TreeClassifierBase):
    _tree_type = "oblivious"


class CARTRegressor(_TreeRegressorBase):
    _tree_type = "cart"


class ExtraTreeRegressor(_TreeRegressorBase):
    _tree_type = "randomized"


class ObliviousTreeRegressor(_TreeRegressorBase):
    _tree_type = "oblivious"


__all__ = [
    "ID3Classifier",
    "C45Classifier",
    "CARTClassifier",
    "ExtraTreeClassifier",
    "ObliviousTreeClassifier",
    "CARTRegressor",
    "ExtraTreeRegressor",
    "ObliviousTreeRegressor",
]
