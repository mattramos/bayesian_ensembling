import json
import pprint
from dataclasses import dataclass


@dataclass(repr=False)
class Parameters:
    def to_dict(self):
        return self.__dict__


@dataclass(repr=False)
class GPRParameters(Parameters):
    batch_size: int = 100
    optim_nits: int = 500
