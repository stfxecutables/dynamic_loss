import torch
from torch import Tensor
from torch.nn import Module
from torchvision.transforms import Compose, RandomHorizontalFlip
from torchvision.transforms.autoaugment import TrivialAugmentWide


class Cutout(Module):
    def __init__(self, size: int = 16) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: Tensor) -> Tensor:
        h_max, w_max = x.shape[1:]
        p = self.size
        h = torch.randint(0, h_max - 1, [1]).item()
        w = torch.randint(0, w_max - 1, [1]).item()
        h_stop = min(h + p, h_max)
        w_stop = min(w + p, w_max)
        img = x.clone()
        img[:, h:h_stop, w:w_stop] = 0
        return img


class Augmenter(Module):
    def __init__(self) -> None:
        super().__init__()
        self.augment = Compose([RandomHorizontalFlip(), TrivialAugmentWide(), Cutout()])

    def forward(self, x: Tensor) -> Tensor:
        return self.augment(x)
