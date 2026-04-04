from ._sklearn import _GBMClassifierBase, _GBMRegressorBase


class CARTGBMClassifier(_GBMClassifierBase):
    _tree_type = "cart"


class ExtraGBMClassifier(_GBMClassifierBase):
    _tree_type = "randomized"


class ObliviousGBMClassifier(_GBMClassifierBase):
    _tree_type = "oblivious"


class CARTGBMRegressor(_GBMRegressorBase):
    _tree_type = "cart"


class ExtraGBMRegressor(_GBMRegressorBase):
    _tree_type = "randomized"


class ObliviousGBMRegressor(_GBMRegressorBase):
    _tree_type = "oblivious"


__all__ = [
    "CARTGBMClassifier",
    "ExtraGBMClassifier",
    "ObliviousGBMClassifier",
    "CARTGBMRegressor",
    "ExtraGBMRegressor",
    "ObliviousGBMRegressor",
]
