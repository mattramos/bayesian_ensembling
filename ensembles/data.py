import tensorflow as tf
import typing as tp
from dataclasses import dataclass

from .array_types import ColumnVector, Matrix


@dataclass
class Dataset:
    Xs: tp.List[Matrix]
    y: ColumnVector

    def __post_init__(self):
        assert isinstance(self.Xs, list), "Input data must be a list"
        for dataset in self.Xs:
            tf.debugging.assert_shapes(
                [(dataset, ("N", "D")), (self.y, ("N", 1))]
            )

    @property
    def n_datasets(self) -> int:
        return len(self.Xs)

    @property
    def n(self) -> int:
        return self.y.shape[0]

    def __len__(self) -> int:
        return self.y.shape[0]
