import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data_preprocessing import build_preprocessor
from src.features import add_features


def build_pipeline(model, use_extra_features: bool = False, log_target: bool = False):
    """raw DataFrame -> (optional feature engineering) -> encoding -> model.
    With log_target=True the model learns log(price) and predictions are
    converted back to the original price scale automatically."""
    steps = []
    if use_extra_features:
        steps.append(("features", FunctionTransformer(add_features)))
    steps.append(("prep", build_preprocessor(use_extra_features)))
    steps.append(("model", model))
    pipe = Pipeline(steps)
    if log_target:
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


def param_prefix(log_target: bool) -> str:
    """Prefix to address the model's params from the outside, e.g. 'model__max_depth'."""
    return "regressor__model__" if log_target else "model__"
