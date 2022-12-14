from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import json
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.constants import LOG_ROOT_DIR, OPTIMAL_LR, OPTIMAL_WD
from src.enumerables import Experiment, FusionMethod, Loss, TrainingSubset, VisionDataset
from src.parsing import float_or_none, int_or_none, to_bool


@dataclass
class Config:
    # Experiment args
    vision_dataset: VisionDataset
    experiment: Experiment
    fusion: FusionMethod | None
    subset: TrainingSubset
    ensemble_idx: int | None
    binary: bool = False
    loss: Loss = Loss.CrossEntropy
    loss_threshold: float | None = None
    soften: bool = False
    dyn_epoch: int = -1
    pooled: bool = False
    shuffled: bool = False
    num_classes: int = 10
    # Training args
    augment: bool = True
    resize: int | None = None
    max_epochs: int = 30
    lr_init: float = OPTIMAL_LR
    weight_decay: float = OPTIMAL_WD
    # Loader args
    batch_size: int = 128
    num_workers: int = 4

    @staticmethod
    def from_args(argstr: str | None = None) -> tuple[Config, ArgumentParser]:
        parser = Config.parser()
        if argstr is None:
            args, remain = parser.parse_known_args()
        else:
            args, remain = parser.parse_known_args(argstr.split(" "))
        remain.extend(["--max_epochs", str(args.max_epochs)])
        vision_dataset = args.dataset
        if args.shuffled and args.pooled:
            raise ValueError("Cannot both shuffle and pool ensembles.")
        if args.binary:
            num_classes: int = 2
        else:
            num_classes = vision_dataset.num_classes()
        return (
            Config(
                vision_dataset=vision_dataset,
                experiment=args.experiment,
                fusion=args.fusion,
                subset=args.subset,
                ensemble_idx=args.ensemble_idx,
                binary=args.binary,
                loss=args.loss,
                loss_threshold=args.loss_threshold,
                soften=args.soften,
                dyn_epoch=args.dyn_epoch,
                pooled=args.pooled,
                shuffled=args.shuffled,
                augment=args.augment,
                resize=args.resize,
                lr_init=args.lr,
                weight_decay=args.wd,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                num_classes=num_classes,
                num_workers=args.num_workers,
            ),
            remain,
        )

    @staticmethod
    def parser() -> ArgumentParser:
        p = ArgumentParser()
        p.add_argument(
            "--experiment",
            "--exp",
            type=Experiment.parse,
            help=Experiment.choices(),
            default=Experiment.BaseEnsemble,
        )
        p.add_argument(
            "--subset",
            "--training-subset",
            "--training_subset",
            type=TrainingSubset.parse,
            help=TrainingSubset.choices(),
            default=TrainingSubset.Full,
        )
        p.add_argument(
            "--ensemble_idx",
            "--ensemble-idx",
            "--idx",
            type=int_or_none,
            help="Ensemble index for training of ensembles. None for final runs.",
            default=None,
        )
        p.add_argument(
            "--fusion",
            type=FusionMethod.parseN,
            help=FusionMethod.choicesN(),
            default=None,
        )
        p.add_argument(
            "--binary",
            action="store_true",
            help="Whether to solve a binarized (two-class) problem",
        )
        p.add_argument(
            "--dataset",
            "--data",
            type=VisionDataset.parse,
            help=VisionDataset.choices(),
            default=VisionDataset.CIFAR10,
        )
        p.add_argument(
            "--loss",
            type=Loss.parse,
            help=Loss.choices(),
            default=Loss.CrossEntropy,
        )
        p.add_argument(
            "--loss_threshold",
            "--loss-threshold",
            type=float_or_none,
            help="Threshold in (0, 1] for dynamic loss",
            default=None,
        )
        p.add_argument(
            "--soften",
            action="store_true",
            help="Whether to soften the dynamic loss",
        )
        p.add_argument(
            "--pooled",
            action="store_true",
            help="Whether to pool ensemble raw predictions",
        )
        p.add_argument(
            "--dyn_epoch",
            "--dyn-epoch",
            type=int,
            help="At what epoch to switch to dynamic loss",
            default=-1,
        )
        p.add_argument(
            "--shuffled",
            action="store_true",
            help="Whether to shuffle ensemble ordering",
        )
        p.add_argument(
            "--augment",
            "--augmentation",
            type=to_bool,
            help="Whether or not to use full augmentation",
            default=True,
        )
        p.add_argument(
            "--resize",
            type=int_or_none,
            help="Image size to use for training and testing.",
            default=None,
        )
        p.add_argument(
            "--lr",
            "--lr_init",
            type=float_or_none,
            help=f"Initial learning rate. Default={OPTIMAL_LR:1.1e}.",
            default=OPTIMAL_LR,
        )
        p.add_argument(
            "--wd",
            "--weight_decay",
            type=float_or_none,
            help=f"Weight decay. Default={OPTIMAL_WD:1.1e}.",
            default=OPTIMAL_WD,
        )
        p.add_argument(
            "--max_epochs",
            type=int_or_none,
            help="Number of epochs.",
            default=30,
        )
        p.add_argument(
            "--batch_size",
            type=int_or_none,
            help="Batch size. Default 128.",
            default=128,
        )
        p.add_argument(
            "--num_workers",
            type=int_or_none,
            help="Number of workers for dataloading.  Default 1.",
            default=1,
        )
        return p

    def log_base_dir(self) -> Path:
        """Directory is:
        logs/[tune]/[experiment]/[subset]/[fusion]/[data]/[binary]/[augment]
        """
        e = self.experiment.value
        s = self.subset.value
        i = f"/idx_{self.ensemble_idx}" if self.fusion is None else ""
        f = self.fusion.value if self.fusion is not None else "none"
        d = self.vision_dataset.value
        b = "binary" if self.binary else "all-classes"
        a = "augmented" if self.augment else "no-augment"
        r = "no-resize" if self.resize is None else f"{self.resize}x{self.resize}"
        if (not self.shuffled) and (not self.pooled):
            p = "/no-shuffle"
        elif self.shuffled:
            p = "/shuffled"
        elif self.pooled:
            p = "/pooled"
        else:
            raise ValueError("Impossible!")
        if self.fusion is None:
            p = ""

        outdir = LOG_ROOT_DIR / f"{e}/{s}{i}/{f}{p}/{d}/{b}/{a}/{r}"
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir

    def loggable(self) -> dict[str, Any]:
        a = "augmented" if self.augment else "no-augment"
        return dict(
            augment=a,
            pooled=self.pooled,
            shuffled_ensembles=self.shuffled,
            soften=self.soften,
            lr_init=self.lr_init,
            wd=self.weight_decay,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
        )

    def to_json(self, log_version_dir: Path) -> None:
        logdir = log_version_dir.resolve()
        if not logdir.exists():
            raise FileNotFoundError(f"Log version directory {logdir} does not exist.")
        configs = logdir / "configs"
        configs.mkdir(exist_ok=True, parents=True)
        outfile = configs / "config.json"
        with open(outfile, "w") as handle:
            json.dump(self.__dict__, handle, default=str, indent=2)
        print(f"Saved experiment configuration to {outfile}")

    @staticmethod
    def from_json(log_version_dir: Path) -> Config:
        def eval_(o: Any) -> Any:
            if o is None:
                return None
            return eval(o)

        logdir = log_version_dir.resolve()
        if not logdir.exists():
            raise FileNotFoundError(f"Log version directory {logdir} does not exist.")
        configs = logdir / "configs"
        jsonfile = configs / "config.json"
        with open(jsonfile, "r") as handle:
            config = Namespace(**json.load(handle))
        # handle potentially missing attributes here...
        if hasattr(config, "loss_threshold"):
            threshold = config.loss_threshold
        else:
            threshold = None
        return Config(
            vision_dataset=eval_(config.vision_dataset),
            experiment=eval_(config.experiment),
            fusion=eval_(config.fusion),
            subset=eval_(config.subset),
            ensemble_idx=int(config.ensemble_idx),
            binary=bool(config.binary),
            loss=eval_(config.loss),
            loss_threshold=threshold,
            augment=bool(config.augment),
            resize=int(config.resize),
            lr_init=float(config.lr_init),
            weight_decay=float(config.weight_decay),
            max_epochs=int(config.max_epochs),
            batch_size=int(config.batch_size),
            num_classes=int(config.num_classes),
            num_workers=int(config.num_workers),
        )

    def __str__(self) -> str:
        fmt = []
        for field, value in self.__dict__.items():
            val = value.name if isinstance(value, Enum) else value
            fmt.append(f"{field}={val}")
        inner = ",".join(fmt)
        return f"{self.__class__.__name__}({inner})"