import typing as tp
from dataclasses import dataclass

from .array_types import ColumnVector, Matrix


@dataclass
class Dataset:
    Xs: tp.List[Matrix]
    y: ColumnVector

    @property
    def n_datasets(self) -> int:
        return len(self.Xs)
