import typing as tp

import tensorflow as tf

from .data import ModelCollection, ProcessModel
from .weights import Weight
from .models import Model


class Ensemble:
    def __init__(
        self, models: ModelCollection, observations: ProcessModel, name: "Ensemble"
    ) -> None:
        self.models = models
        self.observations = observations
        self.name = name
        self.models: tp.List = None
        assert self.validate_inputs()

    def validate_inputs(self) -> bool:
        equality = [len(m) == len(self.observations) for m in self.models]
        return all(equality)

    def fit(self, weighting_scheme: Weight) -> None:
        for t in time_points:
            weights = compute_weights(self.models, self.observations)
