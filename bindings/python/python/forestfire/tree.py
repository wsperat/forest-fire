from ._sklearn import _TreeClassifierBase, _TreeRegressorBase


class ID3Classifier(_TreeClassifierBase):
    _tree_type = "id3"


class C45Classifier(_TreeClassifierBase):
    _tree_type = "c45"


class CARTClassifier(_TreeClassifierBase):
    _tree_type = "cart"


class ExtraClassifier(_TreeClassifierBase):
    _tree_type = "randomized"


class ObliviousClassifier(_TreeClassifierBase):
    _tree_type = "oblivious"


class CARTRegressor(_TreeRegressorBase):
    _tree_type = "cart"


class ExtraRegressor(_TreeRegressorBase):
    _tree_type = "randomized"


class ObliviousRegressor(_TreeRegressorBase):
    _tree_type = "oblivious"


__all__ = [
    "ID3Classifier",
    "C45Classifier",
    "CARTClassifier",
    "ExtraClassifier",
    "ObliviousClassifier",
    "CARTRegressor",
    "ExtraRegressor",
    "ObliviousRegressor",
]
