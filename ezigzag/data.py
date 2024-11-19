"""
ezigzag.data module

@author: phdenzel
"""
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as T
from tqdm import tqdm
from typing import Optional, Any, Callable, Iterator


class TrainTestDataLoader:
    """DataLoader wrapper for train and test datasets"""
    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        mode: str = "train",
        train_kwargs: Optional[dict] = None,
        eval_kwargs: Optional[dict] = None,
        test_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Constructor

        Args:
          train_dataset (Dataset): Train dataset
          test_dataset (Dataset): Test dataset
          eval_dataset (Dataset): Eval dataset
          mode (str): Loader mode, either 'train', 'test', or 'eval'
          train_kwargs (dict): Keyword arguments for the train dataloader initialization.
          eval_kwargs (dict): Keyword arguments for the eval dataloader initialization.
          test_kwargs (dict): Keyword arguments for the test dataloader initialization.
          **kwargs (**dict): torch.utils.data.DataLoader keyword arguments
        """
        self.mode = mode
        self.datasets: list[Dataset] = [train_dataset, test_dataset]
        if eval_dataset is not None:
            self.datasets.append(eval_dataset)
        # Dataloader(s)
        kwargs.setdefault("batch_size", 1)
        kwargs.setdefault("shuffle", True)
        kwargs.setdefault("pin_memory", False)
        self.dataloader_kwargs = [
            kwargs if train_kwargs is None else kwargs | train_kwargs,
            kwargs if eval_kwargs is None else kwargs | eval_kwargs,
            kwargs if test_kwargs is None else kwargs | test_kwargs,
        ]
        self.dataloaders: list[DataLoader] = [
            DataLoader(ds, **{
                k: v for k, v in kw.items()
                if k in DataLoader.__init__.__annotations__
            })
            for ds, kw in zip(self.datasets, self.dataloader_kwargs)
        ]

    def train(self):
        """Set dataloader mode to train"""
        self.mode = "train"

    def eval(self):
        """Set dataloader mode to eval"""
        self.mode = "eval"

    def test(self):
        """Set dataloader mode to test"""
        self.mode = "test"

    @property
    def is_on_train(self) -> bool:
        """Boolean for dataloader mode == 'train'"""
        return self.mode == "train"

    @property
    def is_on_eval(self) -> bool:
        """Boolean for dataloader mode == 'eval'"""
        return self.mode in "eval"

    @property
    def is_on_test(self) -> bool:
        """Boolean for dataloader mode == 'test'"""
        return self.mode == "test"

    @property
    def iterator(self) -> DataLoader:
        """Mode-dependent fetching of the dataloader"""
        if self.is_on_train:
            return self.dataloaders[0]
        elif self.is_on_test and self.dataloaders[1] is not None:
            return self.dataloaders[1]
        elif self.is_on_eval and self.dataloaders[-1] is not None:
            return self.dataloaders[-1]
        return self.dataloaders[0]

    @property
    def dataset(self) -> Dataset:
        """Current dataset"""
        if self.is_on_train:
            return self.datasets[0]
        elif self.is_on_test:
            return self.datasets[1]
        elif self.is_on_eval:
            return self.datasets[-1]
        return self.datasets[0]

    @property
    def batch_size(self) -> int:
        """Batch size of the dataloader"""
        if isinstance(self.iterator.batch_size, int):
            return int(self.iterator.batch_size)
        return 1

    def __len__(self) -> int:
        """Number of samples in the current dataset"""
        return int(self.dataset.__len__())

    def __iter__(self) -> Iterator:
        """Standard iterator"""
        for data in self.iterator:
            yield data

    def tqdm(self, iter_fn: Optional[Callable] = enumerate, **kwargs) -> Any:
        """A tqdm iterator"""
        if iter_fn is not None:
            dl_iter = iter_fn(self.iterator)
        else:
            dl_iter = self.iterator
        return tqdm(dl_iter, total=len(self)//self.batch_size, **kwargs)


def init_dataloader(
    data_dir: str | Path,
    transforms: Optional[T.Transform | list[T.Transform]] = None,
    dtype: torch.dtype = torch.float32,
    **kwargs
) -> TrainTestDataLoader:
    """
    Initialize dataloader of image datasets

    Args:
      train_dir (str | Path): Image root folder containing train data
      test_dir (str | Path): Image root folder containing test data
      transforms (Transform | list[Transform]): Transforms
      dtype (torch.dtype): Data format; default: torch.float32
      kwargs (dict): Keyword arguments for torch.utils.data.DataLoader
    """
    train_dir = Path(data_dir) / "dataset"
    test_dir = Path(data_dir) / "testset"
    if transforms is None:
        transform = T.Compose([T.ToImage(), T.ToDtype(dtype, scale=True)])
    elif isinstance(transforms, list | tuple):
        transform = T.Compose(transforms)
    else:
        transform = transforms
    if Path(train_dir).exists():
        train_ds = ImageFolder(train_dir, transform=transform)
    if test_dir is not None and Path(test_dir).exists():
        test_ds = ImageFolder(test_dir, transform=transform)
    else:
        test_ds = None
    return TrainTestDataLoader(train_ds, test_ds, **kwargs)


def jellyfish_dataloader(
    data_dir: Path = Path.home() / "ezigzag/data/jellyfish",
    **kwargs,
):
    """Initialize the jellyfish dataloader"""
    if not Path(data_dir).exists():
        print("Download the dataset first with `cd data && FTPPASS='...' ./download_jellyfish.sh`")
        exit(0)
    kwargs['transforms'] = [
        T.Resize(256),
        T.CenterCrop(256),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ]
    return init_dataloader(data_dir, **kwargs)


def simpsons_dataloader(
    data_dir: Path = Path.home() / "ezigzag/data/simpsons",
    **kwargs,
):
    """Initialize the simpsons dataloader"""
    if not Path(data_dir).exists():
        print("Download the dataset first with `cd data && FTPPASS='...' ./download_simpsons.sh`")
        exit(0)
    kwargs['transforms'] = [
        T.Resize(256),
        T.CenterCrop(256),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        # T.Normalize(mean=(0.4146,), std=(0.0012,)),
    ]
    return init_dataloader(data_dir, **kwargs)


def xray_dataloader(
    data_dir: Path = Path.home() / "ezigzag/data/xray",
    **kwargs,
):
    """Initialize the xray dataloader"""
    if not Path(data_dir).exists():
        print("Download the dataset first with `cd data && FTPPASS='...' ./download_xray.sh`")
        exit(0)
    kwargs['transforms'] = [
        T.Resize(256),
        T.CenterCrop(256),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ]
    return init_dataloader(data_dir, **kwargs)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    dl = simpsons_dataloader()

    for i, batch in dl.tqdm():
        if i in [1, 100, 1000, 4000, 6000]:
            img = batch[0][0].permute(1, 2, 0)
            plt.imshow(img.numpy())
            plt.show()
    
    
