"""
ezigzag.utils module

Contains utility functions

@author: phdenzel
"""
from pathlib import Path
import uuid
import lzma
import base64
import csv
import json
import torch
from torch import nn
from typing import Optional, Any
import ezigzag


def get_run_id(length: int = 8) -> str:
    """Fetch a run-specific identifier"""
    return str(ezigzag.RUN_UID).replace("-", "")[:length]


def set_run_id(run_id: Optional[uuid.UUID] = None):
    """Set the run-specific identifier"""
    if run_id is None:
        run_id = uuid.uuid4()
    ezigzag.RUN_UID = run_id


def compress_encode(string: str):
    """Compress and encode a string for shorter representations"""
    compressed_data = lzma.compress(string.encode('utf-8'))
    encoded_data = base64.b64encode(compressed_data)
    return encoded_data.decode('utf-8')


def extract_decode(string: str):
    """Decompress and decode a short representation string"""
    compressed_data = base64.b64decode(string)
    original_data = lzma.decompress(compressed_data)
    return original_data.decode('utf-8')


def save_config(
    config: dict,
    path: str | Path,
    include_git_state: bool = False,
) -> Path:
    """
    Save the configuration in a JSON file

    Args:
      config (dict): Dictionary containing the entire run configuration
      path (str | Path): Path to the checkpoint directory or file
      include_git_state (bool): Include git state in the configuration
    """
    path = Path(path)
    if path.is_dir():
        path = path / f"{get_run_id()}.json"
    elif path.suffix != ".json":
        path = path.with_suffix(".json")
    config["run_uid"] = str(ezigzag.RUN_UID)
    if include_git_state:
        config["git_state"] = ezigzag.GIT_STATE
        config["git_diff"] = [compress_encode(d) for d in ezigzag.GIT_DIFF]
    with path.open("w") as f:
        json_dict = {}
        for k, v in config.items():
            if isinstance(v, Path) or isinstance(v, torch.device):
                json_dict[k] = str(v)
            else:
                json_dict[k] = v
        json.dump(json_dict, f, indent=4)
    return path


def load_config(
    filename: str | Path,
    write_globals: bool = True,
):
    """
    Load the configuration from a JSON file

    Args:
      filename (str | Path): Path to the configuration file
      write_globals (bool): Overwrite global runtime variables
    """
    with Path(filename).open("r") as f:
        json_dict = json.load(f)
        if write_globals and "run_uid" in json_dict:
            ezigzag.RUN_UID = uuid.UUID(json_dict["run_uid"])
        if write_globals and "git_state" in json_dict:
            ezigzag.GIT_STATE = json_dict["git_state"]
        if write_globals and "git_diff" in json_dict:
            ezigzag.GIT_DIFF = [extract_decode(d) for d in json_dict["git_diff"]]
        for k in json_dict:
            if k.endswith("_dir") or k == "root":
                json_dict[k] = Path(json_dict[k])
    return json_dict


def write_csv_log(
    data: dict,
    filename: Optional[str | Path] = None,
    _file: Optional[Any] = None,
    write_row: bool = True,
    write_header: bool = False,
) -> Any:
    """
    Write to a CSV log file

    Args:
      data (dict): data which to write to file
      filename (str | Path): the filename of the csv file
      _file (io.TextIOWrapper): an opened IO file buffer into which to stream the log data
      write_row (bool): if True, the data dictionary is written to the csv file
      write_header (bool): if True, a header is written to the csv file (instead of data)
    """
    if _file is None and filename is None:
        raise ValueError("Need at least a filename reference for writing data to file!")
    elif _file is None:
        filename = Path(filename)
        if filename.is_dir():
            filename = filename / f"{get_run_id()}"
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True)
        _file = filename.open(mode="a+")
    elif _file.closed:
        _file.open(mode="a+")
    writer = csv.DictWriter(_file, fieldnames=data.keys())
    if write_header:
        writer.writeheader()
    elif write_row:
        writer.writerow(data)
    return _file


def write_ckpt(
    filepath: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    loss: Optional[float] = None,
    epoch: Optional[int | str] = None,
    as_latest: bool = False,
) -> Path:
    """
    Save a checkpoint including model, optimizer, and scheduler states

    Args:
      filepath (str | Path): The filepath where to save the checkpoint
      model (nn.Module): A torch model
      optimizer (torch.optim.Optimizer): A torch optimizer
      scheduler (torch.optim.lr_scheduler.LRScheduler):
        A torch learning rate scheduler
      loss (float): The latest calculated loss
      epoch (int | str): The current epoch
      as_latest (bool): If True, the checkpoint is labelled as 'epoch_latest.pth'
    """
    run_id = get_run_id()
    # default epoch and epoch filename string
    if epoch is None:
        epoch = epoch_str = "latest"
    elif isinstance(epoch, int):
        epoch_str = f"{epoch:04d}" if epoch >= 0 else "init"
    else:
        epoch_str = str(epoch)
    if as_latest:
        epoch_str = "latest"
    # default filename
    filename = Path(filepath)
    if filename.is_dir() or filename.suffix not in (".pth", ".pt"):
        filename = filename / f"{run_id}_epoch_{epoch_str}.pth"
    # compile state_dict
    state: dict = {}
    state["epoch"] = epoch
    state["model_state_dict"] = model.state_dict()
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if loss is not None:
        state["loss"] = loss
    torch.save(state, filename)
    return filename


def load_ckpt(
    filepath: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    loss: Optional[float] = None,
    epoch: Optional[int | str] = None,
) -> tuple[int, float | None]:
    """
    Load a checkpoint including model, optimizer, and scheduler states

    Args:
      filepath (str | Path): The filepath where to save the checkpoint
      model (nn.Module): A torch model
      optimizer (torch.optim.Optimizer): A torch optimizer
      scheduler (torch.optim.lr_scheduler.LRScheduler):
        A torch learning rate scheduler
      loss (float): The latest calculated loss
      epoch (int | str): If the file
    """
    run_id = get_run_id()
    filename = Path(filepath)
    if filename.is_dir():
        if epoch is not None:
            epoch_str: str = str(epoch)
            if isinstance(epoch, int):
                epoch_str = f"{epoch::04d}"
            filename = filename / f"{run_id}_epoch_{epoch_str}.pth"
        else:
            filename = filename / f"{run_id}_epoch_latest.pth"
    state = torch.load(filename, weights_only=True)
    ckpt_epoch = int(state["epoch"])
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if "loss" in state:
        loss = state["loss"]
    return ckpt_epoch, loss
