import json
import pprint
from dataclasses import dataclass, field


@dataclass(repr=False)
class Parameters:
    def to_dict(self):
        return self.__dict__


@dataclass(repr=False)
class GPRParameters(Parameters):
    batch_size: int = 100
    optim_nits: int = 500


@dataclass(repr=False)
class SGPRParameters(Parameters):
    batch_size: int = 100
    optim_nits: int = 2500
    log_interval: int = 20
    n_inducing: int = 50
    learning_rate: float = 0.01
