"""
ezigzag module

@author: phdenzel
"""
from pathlib import Path
from argparse import ArgumentParser
import uuid
import git
import torch
from torch import nn
from ezigzag.utils import get_run_id
from typing import Any


__version__ = "0.0.1.dev1"


RUN_UID = uuid.uuid4()
repository = git.Repo(search_parent_directories=True)
GIT_STATE = repository.head.object.hexsha
GIT_DIFF = [
    str(diff)
    for diff in
    repository.index.diff(None, create_patch=True)
    + repository.index.diff("HEAD", create_patch=True)
]


def configure_torch_backends(
    empty_cache: bool = True,
    verbose: bool = False
):
    """
    Configure torch backends, optimize algorithms, and empty cache
    """
    if torch.backends.cuda.is_built():
        if verbose:
            print(f"cuDNN backend enabled: {torch.backends.cudnn.enabled}")
        # auto-tune convolution algorithms
        torch.backends.cudnn.benchmark = True
        if empty_cache:
            torch.cuda.empty_cache()
            if verbose:
                print("Released unoccupied memory in cache [CUDA]")
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        if verbose:
            print(f"MPS backend available: {torch.backends.mps.is_available()}")
        if empty_cache:
            torch.mps.empty_cache()
            if verbose:
                print("Released unoccupied memory in cache [MPS]")


def parse_args(**kwargs) -> dict:
    """Parse arguments"""
    parser = ArgumentParser()

    # General arguments
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Set verbosity."
    )
    parser.add_argument(
        "-d", "--device",
        type=torch.device,
        default=kwargs.get("device", torch.device("cpu")),
        help="Device for computation, e.g. 'cpu', 'cuda', or 'mps'."
    )

    # Data configuration
    parser.add_argument(
        "-f", "--root", "--file",
        type=Path,
        default=kwargs.get("root", None),
        help="Root of the dataset directory"
    )
    parser.add_argument(
        "--split",
        type=lambda s: [float(item) for item in s.split(',')],
        default=kwargs.get("split", [0.9, 0.1]),
        help="Dataset split ratios (need to add up to 1)"
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=kwargs.get("batch_size", 1),
        help="Batch size for the dataloader."
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=kwargs.get("shuffle", True),
        help="Shuffle the dataset."
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_false",
        dest="shuffle",
        help="Do not shuffle the dataset"
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=kwargs.get("pin_memory", True),
        help="Use memory pinning during dataloading"
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_false",
        dest="pin_memory",
        help="Do not use memory pinning during dataloading"
    )

    # Model configuration
    parser.add_argument(
        "--features",
        type=lambda s: [int(f) for f in s.split(',')],
        default=kwargs.get("features", [512, 256, 128]),
        help="Number of output features for each hidden layer"
    )
    parser.add_argument(
        "--activations",
        type=lambda s: s.split(','),
        default=kwargs.get("activations", "silu"),
        help="Activation name(s) for each MLP layer"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=kwargs.get("dropout", 0),
        help="Dropout rate in between each layer"
    )

    # Optimizer configuration
    parser.add_argument(
        "--optimizer", "--opt",
        type=str,
        default=kwargs.get("optimizer", "Adam"),
        help="The type of optimizer."
    )
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=kwargs.get("learning_rate", 1e-4),
        help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--lr-decay", "--learning-rate-decay",
        type=float,
        default=kwargs.get("lr_decay", 0.1),
        help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--lr-step", "--lr-decay-step",
        type=int,
        default=kwargs.get("lr_step", 20),
        help="Learning rate for the optimizer."
    )

    # Training configuration
    parser.add_argument(
        "-c", "--ckpt-dir", "--checkpoint-dir", "--ckpt-directory", "--checkpoint-directory",
        type=Path,
        default=kwargs.get("ckpt_dir", Path("checkpoints")),
        help="The directory path where checkpoints are saved."
    )
    parser.add_argument(
        "--results-dir", "--results-directory",
        type=Path,
        default=kwargs.get("results_dir", Path("results")),
        help="The directory path where results are saved."
    )
    parser.add_argument(
        "--train-log", "--train-file", "--f-train",
        type=Path,
        default=kwargs.get("train_log", Path(f"{get_run_id()}_losses.csv")),
        help="The file path where training results are logged."
    )
    parser.add_argument(
        "--loss", "--losses", "--loss-functions",
        type=lambda s: s.split(','),
        default=kwargs.get("loss", ["MSE"]),
        help="Use MSE loss for training"
    )
    parser.add_argument(
        "--loss-weights", "--loss-weighting",
        type=lambda s: [float(item) for item in s.split(',')],
        default=kwargs.get("loss_weights", None),
        help="Use KLDiv loss for training"
    )
    parser.add_argument(
        "--epoch_start",
        type=int,
        default=kwargs.get("epoch_start", 0),
        help="Index of starting epoch"
    )
    parser.add_argument(
        "-e", "--n_epochs", "--epochs",
        type=int,
        default=kwargs.get("n_epochs", 100),
        help="Index of starting epoch"
    )
    parser.add_argument(
        "--freq-ckpt", "--freq-checkpoints", "--t-ckpt", "--t-checkpoints",
        type=int,
        default=kwargs.get("freq_ckpt", 10),
        help="Frequency with which to print to stdout [epochs]"
    )
    parser.add_argument(
        "--freq-stdout", "--t-stdout",
        type=int,
        default=kwargs.get("freq_stdout", 25),
        help="Frequency with which to print to stdout [iterations]"
    )
    parser.add_argument(
        "--freq-log", "--t-log",
        type=int,
        default=kwargs.get("freq_log", 75),
        help="Frequency with which to log iterations [iterations]"
    )

    # Evaluation configuration
    parser.add_argument(
        "--eval-log", "--eval-file", "--f-eval",
        type=Path,
        default=kwargs.get("eval_log", Path(f"{get_run_id()}_metrics.csv")),
        help="The file path where evaluation results are logged."
    )
    parser.add_argument(
        "--metrics", "--eval-metrics",
        type=lambda s: s.split(','),
        default=kwargs.get("metrics", ["MSE", "R2"]),
        help="Transform targets from the dataset"
    )

    # Parse arguments
    args, _ = parser.parse_known_args()
    configs = vars(args)
    return configs


def init_optimizer(
    optimizer_cls: str | type[torch.optim.Optimizer],
    model: nn.Module,
    *args,
    **kwargs
) -> Any:
    """
    Initialize an optimizer by name

    Args:
      optimizer_cls (str | type[Optimizer]): The optimizer class as name or class
      model (Module): Module to be optimized
      args: Additional arguments
      kwargs: Additional keyword arguments
    """
    if isinstance(optimizer_cls, str):
        cls = getattr(torch.optim, optimizer_cls)
    else:
        cls = optimizer_cls
    defaults = dict(lr=kwargs.get("learning_rate", 1e-3))
    for k in cls.__init__.__annotations__:
        if k in kwargs:
            defaults[k] = kwargs[k]
    return cls(model.parameters(), **defaults)


if __name__ == "__main__":

    config = parse_args()
    print(config)
