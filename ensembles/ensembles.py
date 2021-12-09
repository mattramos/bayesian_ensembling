import typing as tp

import tensorflow as tf

from .data import Dataset
from .models import Model


class Ensemble:
    def __init__(self, name: "Ensemble") -> None:
        self.name = name
        self.models: tp.List = None

    def fit(self, data: Dataset, base_model_constructor: tp.Callable, params: dict) -> None:
        y = data.y
        Xs = data.Xs
        self.models = [base_model_constructor() for _ in range(data.n_datasets)]
        [model.fit(X, y, params) for X, model in zip(Xs, self.models)]

    def predict(self, data: Dataset, params: dict) -> tp.List[tp.Tuple[tf.Tensor, tf.Tensor]]:
        Xs = data.Xs
        preds = [model.predict(X, params) for X, model in zip(Xs, self.models)]
        return preds
