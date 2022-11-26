from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from argparse import ArgumentParser
from dataclasses import dataclass

from src.enumerables import Experiment, FusionMethod, TrainingSubset, VisionDataset
from src.parsing import float_or_none, int_or_none, to_bool


@dataclass
class Config:
    vision_dataset: VisionDataset
    experiment: Experiment
    fusion: FusionMethod
    subset: TrainingSubset
    binary: bool = False
    augment: bool = True
    max_epochs: int = 30
    batch_size: int = 32
    lr_init: float = 5e-4
    weight_decay: float = 5e-3
    num_classes: int = 10

    @staticmethod
    def from_args(argstr: str | None = None) -> tuple[Config, ArgumentParser]:
        parser = Config.parser()
        if argstr is None:
            args, remain = parser.parse_known_args()
        else:
            args, remain = parser.parse_known_args(argstr.split(" "))
        remain.extend(["--max_epochs", str(args.max_epochs)])
        vision_dataset = args.vision_dataset
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
                binary=args.binary,
                augment=args.augment,
                lr_init=args.lr,
                weight_decay=args.wd,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                num_classes=num_classes,
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
            type=TrainingSubset.parse,
            help=TrainingSubset.choices(),
            default=TrainingSubset.Full,
        )
        p.add_argument(
            "--fusion",
            type=FusionMethod.parse,
            help=FusionMethod.choices(),
            default=FusionMethod.Vote,
        )
        p.add_argument(
            "--binary",
            action="store_true",
            help="Whether to solve a binarized (two-class) problem",
        )
        p.add_argument(
            "--lr",
            "--lr_init",
            type=float_or_none,
            help="Initial learning rate. Default=5e-4.",
            default=5e-4,
        )
        p.add_argument(
            "--wd",
            "--weight_decay",
            type=float_or_none,
            help="Weight decay. Default=5e-3.",
            default=5e-3,
        )
        p.add_argument(
            "--max_epochs",
            type=int_or_none,
            help="Number of epochs. Should be None unless debugging.",
            default=10,
        )
        p.add_argument(
            "--batch_size",
            type=int_or_none,
            help="Batch size. Default 32.",
            default=32,
        )
        p.add_argument(
            "--augment",
            "--augmentation",
            type=to_bool,
            help="Whether or not to use full augmentation",
            default=True,
        )
        return p

    def log_base_dir(self) -> Path:
        """Directory is:
        logs/[model]/[augmentation]/[patch_size]/
        """
        m = self.model.value
        a = "none" if self.augment is None else self.augment.value
        p = f"{self.patch_size}x{self.patch_size}"
        outdir: Path = LOG_ROOT_DIR / f"{m}/{a}/{p}"
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir

    def loggable(self) -> Dict[str, Any]:
        aug = "none" if self.augment is None else self.augment.value
        max_epochs = self.max_epochs if self.max_epochs is not None else -1
        return dict(
            augmentation=aug,
            lr_init=self.lr_init,
            wd=self.weight_decay,
            max_epochs=max_epochs,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            norm=0 if self.norm is SegNorm.MinMax else 1,
        )

    def to_json(self, log_version_dir: Path) -> None:
        logdir = log_version_dir.resolve()
        if not logdir.exists():
            raise FileNotFoundError(f"Log version directory {logdir} does not exist.")
        configs = logdir / "configs"
        configs.mkdir(exist_ok=True)
        outfile = configs / "exp_config.json"
        with open(outfile, "w") as handle:
            json.dump(self.__dict__, handle, default=str, indent=2)
        print(f"Saved experiment configuration to {outfile}")

    def num_classes(self) -> int:
        return 1 + len(self.classes)  # need to count background...

    def __str__(self) -> str:
        fmt = []
        for field, value in self.__dict__.items():
            val = value.name if isinstance(value, Enum) else value
            fmt.append(f"{field}={val}")
        inner = ",".join(fmt)
        return f"{self.__class__.__name__}({inner})"