from dataclasses import dataclass


@dataclass
class Config:
    max_epochs: int = 30
    lr_init: float = 5e-4
    weight_decay: float = 5e-3