from ._sklearn import _ForestClassifierBase, _ForestRegressorBase


class ID3RandomForestClassifier(_ForestClassifierBase):
    _tree_type = "id3"


class C45RandomForestClassifier(_ForestClassifierBase):
    _tree_type = "c45"


class CARTRandomForestClassifier(_ForestClassifierBase):
    _tree_type = "cart"


class ExtraRandomForestClassifier(_ForestClassifierBase):
    _tree_type = "randomized"


class ObliviousRandomForestClassifier(_ForestClassifierBase):
    _tree_type = "oblivious"


class CARTRandomForestRegressor(_ForestRegressorBase):
    _tree_type = "cart"


class ExtraRandomForestRegressor(_ForestRegressorBase):
    _tree_type = "randomized"


class ObliviousRandomForestRegressor(_ForestRegressorBase):
    _tree_type = "oblivious"


__all__ = [
    "ID3RandomForestClassifier",
    "C45RandomForestClassifier",
    "CARTRandomForestClassifier",
    "ExtraRandomForestClassifier",
    "ObliviousRandomForestClassifier",
    "CARTRandomForestRegressor",
    "ExtraRandomForestRegressor",
    "ObliviousRandomForestRegressor",
]
