"""
ezigzag.data module

@author: phdenzel
"""

from pathlib import Path
import fnmatch
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as T
from tqdm import tqdm
from typing import Optional, Any, Callable, Iterator


class HDF5Dataset(Dataset):
    """Dataset for loading HDF5 frames."""

    def __init__(
        self,
        data_dir: str | Path,
        file_key: str = "**/*.hdf5",
        groups: str | tuple[str, ...] | None = None,
        meta_groups: str | tuple[str, ...] | None = None,
        scheme: str | None = None,
        dim: int = 0,
        n_channels: int | None = None,
        dtype: torch.dtype = torch.float32,
        collate: bool = True,
        preload: bool = False,
        **kwargs,
    ):
        """Constructor.

        Args:
          data_dir (str | Path): Path to the data directory containing the HDF5 files
          file_key (str): Key to filter particular files
          groups (tuple[str, ...]): Filter pattern for HDF5 groups containing datasets
          meta_groups (tuple[str, ...]): Filter pattern for HDF5 attributes containing metadata
          scheme (str): HDF5 metadata loading scheme how HDF5 groups for attributes (metadata) map
            onto HDF5 groups for datasets; can be
            - 'analog': HDF5 datasets and attributes are at the same HDF5 groups
            - 'bijective': HDF5 datasets and attributes are at parallel HDF5 groups
            - 'surjective': HDF5 datasets have per-sample attributes at multiple HDF5 groupse
            - 'collective': HDF5 datasets have all a single HDF5 group of attributes
          dim (int): Dimension along which the HDF5 dataset indexing is performed.
          dtype (torch.dtype): Data format; default: torch.float32
          preload (bool): Preload and cache the dataset.
          sort_key: (Callable): Sorting key for the HDF5 groups.
          kwargs (dict): Keyword arguments for h5py.File.


            driver
              Name of the driver to use.  Legal values are None (default,
              recommended), 'core', 'sec2', 'direct', 'stdio', 'mpio', 'ros3'.
            libver
              Library version bounds.  Supported values: 'earliest', 'v108',
              'v110', 'v112'  and 'latest'.
            userblock_size
                Desired size of user block.  Only allowed when creating a new
                file (mode w, w- or x).
            swmr
                Open the file in SWMR read mode. Only used when mode = 'r'.
            rdcc_nbytes
                Total size of the dataset chunk cache in bytes. The default size
                is 1024**2 (1 MiB) per dataset. Applies to all datasets unless individually changed.
            rdcc_w0
                The chunk preemption policy for all datasets.  This must be
                between 0 and 1 inclusive and indicates the weighting according to
                which chunks which have been fully read or written are penalized
                when determining which chunks to flush from cache.  A value of 0
                means fully read or written chunks are treated no differently than
                other chunks (the preemption is strictly LRU) while a value of 1
                means fully read or written chunks are always preempted before
                other chunks.  If your application only reads or writes data once,
                this can be safely set to 1.  Otherwise, this should be set lower
                depending on how often you re-read or re-write the same data.  The
                default value is 0.75. Applies to all datasets unless individually changed.
                rdcc_nslots
                The number of chunk slots in the raw data chunk cache for this
                file. Increasing this value reduces the number of cache collisions,
                but slightly increases the memory used. Due to the hashing
                strategy, this value should ideally be a prime number. As a rule of
                thumb, this value should be at least 10 times the number of chunks
                that can fit in rdcc_nbytes bytes. For maximum performance, this
                value should be set approximately 100 times that number of
                chunks. The default value is 521. Applies to all datasets unless individually changed.
            track_order
                Track dataset/group/attribute creation order under root group
                if True. If None use global default h5.get_config().track_order.
            fs_strategy
                The file space handling strategy to be used.  Only allowed when
                creating a new file (mode w, w- or x).  Defined as:
                "fsm"        FSM, Aggregators, VFD
                "page"       Paged FSM, VFD
                "aggregate"  Aggregators, VFD
                "none"       VFD
                If None use HDF5 defaults.
            fs_page_size
                File space page size in bytes. Only used when fs_strategy="page". If
                None use the HDF5 default (4096 bytes).
            fs_persist
                A boolean value to indicate whether free space should be persistent
                or not.  Only allowed when creating a new file.  The default value
                is False.
            fs_threshold
                The smallest free-space section size that the free space manager
                will track.  Only allowed when creating a new file.  The default
                value is 1.
            page_buf_size
                Page buffer size in bytes. Only allowed for HDF5 files created with
                fs_strategy="page". Must be a power of two value and greater or
                equal than the file space page size when creating the file. It is
                not used by default.
            min_meta_keep
                Minimum percentage of metadata to keep in the page buffer before
                allowing pages containing metadata to be evicted. Applicable only if
                page_buf_size is set. Default value is zero.
            min_raw_keep
                Minimum percentage of raw data to keep in the page buffer before
                allowing pages containing raw data to be evicted. Applicable only if
                page_buf_size is set. Default value is zero.
            locking
                The file locking behavior. Defined as:

                - False (or "false") --  Disable file locking
                - True (or "true")   --  Enable file locking
                - "best-effort"      --  Enable file locking but ignore some errors
                - None               --  Use HDF5 defaults

                .. warning::

                    The HDF5_USE_FILE_LOCKING environment variable can override
                    this parameter.

                Only available with HDF5 >= 1.12.1 or 1.10.x >= 1.10.7.

            alignment_threshold
                Together with ``alignment_interval``, this property ensures that
                any file object greater than or equal in size to the alignment
                threshold (in bytes) will be aligned on an address which is a
                multiple of alignment interval.

            alignment_interval
                This property should be used in conjunction with
                ``alignment_threshold``. See the description above. For more
                details, see
                https://portal.hdfgroup.org/display/HDF5/H5P_SET_ALIGNMENT

            meta_block_size
                Set the current minimum size, in bytes, of new metadata block allocations.
                See https://portal.hdfgroup.org/display/HDF5/H5P_SET_META_BLOCK_SIZE

            Additional keywords
                Passed on to the selected file driver.
        """
        self.data_dir = Path(data_dir)
        self.files = sorted([f for f in self.data_dir.rglob(file_key) if f.is_file()])
        self.frame_args = kwargs

        self.groups: tuple[str, ...]
        self.meta_groups: tuple[str, ...] | None
        if groups is None:
            self.groups = ("**/images*",)
        else:
            self.groups = (groups,) if isinstance(groups, str) else groups
        if meta_groups is None:
            self.meta_groups = None
            self.scheme = "analog" if scheme is None else scheme
        else:
            self.meta_groups = (meta_groups,) if isinstance(meta_groups, str) else meta_groups
            self.scheme = "surjective" if scheme is None else scheme
        self.n_channels = n_channels
        self.dtype = dtype
        self.collate = collate
        self.preload = preload

        self.load_frame(**self.frame_args)
        self.make_index(dim=dim, scheme=self.scheme, collate=self.collate)

    @staticmethod
    def default_sort_key(x: Any) -> int:
        """Default sort key for HDF5 groups."""
        return int(x.split("/")[-1]) if x.split("/")[-1].isdigit() else 0

    def load_frame(self, **kwargs) -> list[h5py.File] | None:
        """Load HDF5 file instances."""
        if self.files:
            self._frame: list[h5py.File] = [h5py.File(f, "r", **kwargs) for f in self.files]
            return self._frame
        return None

    @property
    def frame(self) -> list[h5py.File]:
        """Lazy-loading list of HDF5 file instances."""
        if not hasattr(self, "_frame"):
            self.load_frame(**self.frame_args)
        return self._frame

    @property
    def frame_structure(self, sort_key: Callable | None = None) -> dict[int, list[str]]:
        """Fetch unfiltered frame structure (HDF5 Groups) of each frame."""
        if sort_key is None:
            sort_key = self.default_sort_key
        if not hasattr(self, "_frame_structure"):
            tree: dict[int, list[str]] = {}
            for i, f in enumerate(self.frame):
                tree[i] = []
                f.visit(tree[i].append)
            self._frame_structure = {
                i: sorted(v, key=sort_key)
                for i, v in tree.items()
            }
        return self._frame_structure

    def _map_analog(
        self, frame_index: int, keys: list[str], dim: int = 0
    ) -> dict[int, tuple[str, str]]:
        """Index datasets using the 'analog' mapping scheme.

        Returns:
          mapping: A dictionary that maps continuous indices to HDF5 groups for
            HDF5 dataset samples and corresponding HDF5 attributes, e.g.
            mapping[42] -> (path/to/images, path/to/metadata_42)
            for which the usage is:

        Example:
          >>> ds = HDF5Dataset("./data", meta_groups="**/metadata/*", scheme="surjective")
          >>> frame = ds.frame[0]
          >>> keys = ["/path/to/dataset"]
          >>> attr_keys = ["/path/to/metadata"]
          >>> mapping = ds._map_surjective(0, keys, attr_keys)[0]
          >>> sample_42 = ds.frame[mapping[42][0]][42]
          >>> metadata_42 = ds.frame[mapping[42][1]].attrs
        """
        mapping: dict[int, tuple[str, str]] = {}
        total_samples = 0
        for key in keys:
            group = self.frame[frame_index][key]
            if isinstance(group, h5py._hl.dataset.Dataset):
                if len(group.shape) == 2:
                    iN = 1
                else:
                    iN = group.shape[dim]
                total_samples += iN
                for i in range(total_samples-iN, total_samples):
                    mapping[i] = (key, key)
        return mapping

    def _map_bijective(self, frame_index: int, ds_keys: list[str], attr_keys: list[str], dim: int = 0):
        """Index datasets using the 'bijective' mapping scheme."""
        return NotImplemented

    def _map_surjective(
        self, frame_index: int, ds_keys: list[str], attr_keys: list[str], dim: int = 0
    ) -> dict[int, tuple[str, str]]:
        """Index datasets using the 'surjective' mapping scheme.

        Returns:
          mapping: A dictionary that maps continuous indices to HDF5 groups for
            HDF5 dataset samples and corresponding HDF5 attributes, e.g.
            mapping[42] -> (path/to/images, path/to/metadata_42).

        Example:
          >>> ds = HDF5Dataset("./data", meta_groups="**/metadata/*", scheme="surjective")
          >>> frame = ds.frame[0]
          >>> keys = ["/path/to/dataset"]
          >>> attr_keys = ["/path/to/metadata"]
          >>> mapping = ds._map_surjective(0, keys, attr_keys)[0]
          >>> sample_42 = ds.frame[mapping[42][0]][42]
          >>> metadata_42 = ds.frame[mapping[42][1]].attrs
        """
        mapping: dict[int, tuple[str, str]] = {}
        total_samples = 0
        for key in ds_keys:
            group = self.frame[frame_index][key]
            if isinstance(group, h5py._hl.dataset.Dataset):
                if len(group.shape) == 2:
                    iN = 1
                else:
                    iN = group.shape[dim]
                total_samples += iN
                if len(attr_keys) != iN:
                    raise ValueError(
                        f"Mismatch in dataset elements ({iN} at {dim=}) and "
                        f"attribute groups ({len(attr_keys)}) for surjective index mapping scheme."
                    )
                for i in range(total_samples-iN, total_samples):
                    mapping[i] = (key, attr_keys[i])
        return mapping

    def _map_collective(self, frame_index: int, ds_keys: int | str, attr_keys: int | str, dim: int = 0):
        """Index datasets using the 'collective' mapping scheme."""
        return NotImplemented

    def make_index(self, dim: int = 0, scheme: str | None = None, collate: bool | None = None):
        """Index the frames for individual samples.

        The index map has the shape:
          [frame index, group index, dataset index]
            or
          [frame index, group index, attribute key]

        Possible scenarios:
          1 frame, 1 group index, 1 dataset
          1 frame, multiple group indices for multiple datasets

        Args:
          dim (int): Dimension across which the datasets are stacked
          scheme (str): HDF5 metadata loading scheme how HDF5 groups for attributes (metadata)
            map onto HDF5 groups for datasets; can be
            - 'analog': HDF5 datasets and attributes are at the same HDF5 groups
            - 'bijective': HDF5 datasets and attributes are at parallel HDF5 groups
            - 'surjective': HDF5 datasets have per-sample attributes at multiple HDF5 groupse
            - 'collective': HDF5 datasets have all a single HDF5 group of attributes
        """
        if scheme is not None:
            self.scheme = scheme
        if collate is not None:
            self.collate = collate
        # filter the frame structures for group keys pointing to datasets and metadata
        groups = {}
        meta_groups = {}
        for i_frame, fnlist in self.frame_structure.items():
            data_keys = [f for g in self.groups for f in fnmatch.filter(fnlist, g)]
            groups[i_frame] = data_keys
            if self.meta_groups:
                meta_keys = [f for g in self.meta_groups for f in fnmatch.filter(fnlist, g)]
                meta_groups[i_frame] = meta_keys
        # map out dataset(s) and metadata
        loc: dict[int, dict] = {}
        for i_frame, data_keys in groups.items():
            if i_frame in meta_groups:
                meta_keys = meta_groups[i_frame]
                match self.scheme:
                    case "analog":
                        mapping = self._map_analog(
                            i_frame, data_keys, dim=dim
                        )
                    case "bijective":
                        mapping = self._map_bijective(
                            i_frame, data_keys, meta_keys, dim=dim
                        )
                    case "surjective":
                        mapping = self._map_surjective(
                            i_frame, data_keys, meta_keys, dim=dim
                        )
                    case "collective":
                        mapping = self._map_collective(
                            i_frame, data_keys, meta_keys, dim=dim
                        )
            else:
                mapping = self._map_analog(i_frame, data_keys, dim=dim)
            if mapping:
                loc[i_frame] = mapping
        # perform indexing across files, and or datasets and metadata
        index: dict[int, list[tuple]] = {}
        if self.collate:
            shift = 0
            for i_frame in loc:
                mapping = loc[i_frame]
                for j in mapping:
                    index[j + shift] = [(i_frame,) + mapping[j]]
                shift += max(loc[i_frame].keys()) + 1
        else:
            for i_frame in loc:
                mapping = loc[i_frame]
                for j in mapping:
                    if j in index:
                        index[j].append((i_frame,) + mapping[j])
                    else:
                        index[j] = [(i_frame,) + mapping[j]]
        self.index = index
        self.length = max(index.keys()) + 1
        return self.index

    def __len__(self):
        """Number of samples in the dataset."""
        return self.length

    def __getitem__(
            self,
            item: int,
    ) -> tuple[torch.Tensor, dict] | tuple[tuple[torch.Tensor, ...], tuple[dict, ...]]:
        """Get samples and metadata at specified index."""
        if item not in self.index:
            raise IndexError(f"{item} not found in index.")

        samples: tuple[torch.Tensor, ...] = ()
        metadata: tuple[dict, ...] = ()
        for loc in self.index[item]:
            # fetch sample
            frame_index, key, attr_key = loc
            sample_arr = self.frame[frame_index][key][frame_index]
            sample = torch.Tensor(sample_arr)
            if len(sample) == 2 or (self.n_channels and sample.shape[0] != self.n_channels):
                sample = sample.view(1, *sample.shape)
            if self.n_channels is None:
                self.n_channels = sample.shape[0]
            samples += (sample,)
            # fetch sample metadata
            info = dict(self.frame[frame_index][attr_key].attrs)
            metadata += (info,)
        if len(samples) > 1:
            return samples, metadata
        else:
            return samples[0], metadata[0]


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
            DataLoader(
                ds,
                **{k: v for k, v in kw.items() if k in DataLoader.__init__.__annotations__},
            )
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
        return tqdm(dl_iter, total=len(self) // self.batch_size, **kwargs)


def init_dataloader(
    data_dir: str | Path,
    transforms: Optional[T.Transform | list[T.Transform]] = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
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
    data_dir: Path = Path(__file__).parent.parent / "ezigzag/data/jellyfish",
    **kwargs,
):
    """Initialize the jellyfish dataloader"""
    if not Path(data_dir).exists():
        print("Download the dataset first with `cd data && FTPPASS='...' ./download_jellyfish.sh`")
        exit(0)
    kwargs["transforms"] = [
        T.Resize(256),
        T.CenterCrop(256),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ]
    return init_dataloader(data_dir, **kwargs)


def simpsons_dataloader(
    data_dir: Path = Path(__file__).parent.parent / "ezigzag/data/simpsons",
    **kwargs,
):
    """Initialize the simpsons dataloader"""
    if not Path(data_dir).exists():
        print("Download the dataset first with `cd data && FTPPASS='...' ./download_simpsons.sh`")
        exit(0)
    kwargs["transforms"] = [
        T.Resize(256),
        T.CenterCrop(256),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        # T.Normalize(mean=(0.4146,), std=(0.0012,)),
    ]
    return init_dataloader(data_dir, **kwargs)


def xray_dataloader(
    data_dir: Path = Path(__file__).parent.parent / "ezigzag/data/xray",
    **kwargs,
):
    """Initialize the xray dataloader"""
    if not Path(data_dir).exists():
        print("Download the dataset first with `cd data && FTPPASS='...' ./download_xray.sh`")
        exit(0)
    kwargs["transforms"] = [
        T.Resize(256),
        T.CenterCrop(256),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ]
    return init_dataloader(data_dir, **kwargs)


def plot_simpson_samples(indices: list[int]):
    """Load the Simpsons dataset and plot selected indices."""
    from matplotlib import pyplot as plt
    dl = simpsons_dataloader()
    for i, batch in dl.tqdm():
        if i in indices:
            img = batch[0][0].permute(1, 2, 0)
            plt.imshow(img.numpy())
            plt.show()


if __name__ == "__main__":
    from ezigzag import parse_args

    config = parse_args()
    config["ckpt_dir"].mkdir(parents=True, exist_ok=True)
    config["results_dir"].mkdir(parents=True, exist_ok=True)

    ds = HDF5Dataset(config["root"], meta_groups="**/metadata/*", collate=True)
    dl = TrainTestDataLoader(ds, ds, batch_size=8)
    for i, batch in dl.tqdm():
        maps = batch[0]
        info = batch[1]
        print(maps.shape, info)
        if i == 0:
            break
