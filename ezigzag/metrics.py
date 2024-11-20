"""
ezigzag.metrics module

Collection of loss functionality and evaluation metrics

@author: phdenzel
"""
import torch
from torch import nn
from torcheval.metrics import (
    MeanSquaredError,
    R2Score,
    PeakSignalNoiseRatio,
    StructuralSimilarity,
    FrechetInceptionDistance,
    Metric
)
from typing import Optional, Any


class ELBO(nn.modules.loss._Loss):
    """Evidence Lower Bound from a Gaussian distribution"""

    def __init__(self, **kwargs):
        """Constructor"""
        super().__init__(**kwargs)

    def forward(self, dist: torch.distributions.MultivariateNormal):
        """Compute the ELBO for a Gaussian distribution"""
        log_var = torch.log(dist.variance)
        return -0.5 * (1 + log_var - dist.mean.pow(2) - log_var.exp()).sum()


class Objective:
    """Objective function composed of various losses"""

    loss_map: dict[str, type[nn.Module]] = {
        "MSE": nn.MSELoss,
        "MAE": nn.L1Loss,
        "BCE": nn.BCELoss,
        "KLDiv": nn.KLDivLoss,
        "ELBO": ELBO,
    }

    loss_kwargs: dict[str, Any] = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": "mean",
        "log_target": False,
    }

    def __init__(
        self,
        loss: Optional[list[str]] = None,
        loss_weights: Optional[list[float]] = None,
        **kwargs: Any
    ):
        """
        Constructor

        Args:
          loss (list[str]): List of losses composing the objective (see below)
          loss_weights (list[float]): Regularization weights for each loss in objective
          **kwargs (dict): For compatibility

        Loss functions:
          MSE: Means Squared Error
          MAE: Mean Average Error
          BCE: Binary Cross-Entropy
          KLDiv: Kullback-Leibler Divergence

        Note:
          Alternative keyword arguments can be:
            {arg}, e.g. MSE
            {arg}_loss
            {arg:lowercase}
            {arg:lowercase}_loss
        """
        self.keys: list[str] = []
        self.values: list[list[torch.Tensor]] = []
        self.loss: torch.Tensor = torch.tensor(0)
        self.weights: list[float] = []
        self.fns: list[nn.Module] = []

        # standardize keys from various inputs
        if loss is None:
            self.keys.append("MSE")
        else:
            for key in self.loss_map.keys():
                if key in loss or \
                   key.lower() in loss or \
                   f"{key}_loss" in loss or \
                   f"{key.lower()}_loss" in loss:
                    self.keys.append(key)
        if not self.keys:
            self.keys.append("MSE")

        # regularization weights for losses
        if loss_weights is None:
            self.weights = len(self.keys)*[1.]
        else:
            self.weights = loss_weights

        # set default keyword arguments for torch loss functions
        for k in kwargs:
            if k in self.loss_kwargs:
                self.loss_kwargs[k] = kwargs[k]

        # torch loss functions
        self.fns = [self.loss_map[k](**self.loss_fn_kwargs[k]) for k in self.keys]
        self.values = [[] for _ in self.keys]

    @property
    def loss_fn_kwargs(self) -> dict[str, dict]:
        """Torch loss function arguments"""
        var_names = {k: v.__init__.__code__.co_varnames[1:] for k, v in self.loss_map.items()}  # noqa
        fn_kwargs = {
            k: {v: self.loss_kwargs[v] for v in var_names[k] if v in self.loss_kwargs}
            for k in self.keys
        }
        if "KLDiv" in self.keys:
            fn_kwargs["KLDiv"]["reduction"] = "batchmean"
        return fn_kwargs

    def __call__(
        self,
        input: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        index: Optional[list[int]] = None
    ) -> torch.Tensor:
        """Compute the objective function"""
        if index is None:
            index = [i for i in range(len(self.fns))]
        elif isinstance(index, int):
            index = [index]
        fns = [self.fns[i] for i in index]
        for i in index:
            self.values[i].clear()
        if target is None:
            for i, fn in zip(index, fns):
                self.values[i].append(fn(input))
        else:
            input = input.squeeze()
            target = target.squeeze()
            for i, fn in zip(index, fns):
                # for log-based terms
                if self.keys[i] == "KLDiv":
                    self.values[i].append(fn(input.log(), target))
                else:
                    self.values[i].append(fn(input, target))
        self.loss = self.calc_objective(index)
        return self.loss

    def calc_objective(
        self,
        index: Optional[list[int]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Calculate objective from pre-calculated losses"""
        if index is None:
            index = [i for i in range(len(self.fns))]
        device = self.values[index[0]][0].device if device is None else device
        dtype = self.values[index[0]][0].dtype if dtype is None else dtype
        weights = [self.weights[i] for i in index]
        values = [self.values[i][0] for i in index]
        objectives = [_lambda * loss for _lambda, loss in zip(weights, values)]
        objective = torch.stack(objectives).sum()
        return objective

    def index(
        self,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
        exclude_loss: bool = False,
    ) -> list[str]:
        """Keys index for results"""
        if prefix is None:
            prefix = ""
        if postfix is None:
            postfix = ""
        idx = []
        if not exclude_loss:
            idx.append(f"{prefix}loss{postfix}")
        return idx+[f"{prefix}{k}{postfix}" for k in self.keys]

    def results(
        self,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None
    ) -> dict[str, float]:
        """Result dictionary of computed objective and each loss component"""
        if prefix is None:
            prefix = ""
        if postfix is None:
            postfix = ""
        results_dict = {f"{prefix}loss{postfix}": self.loss.item()}
        results_dict = results_dict | {
            f"{prefix}{k}{postfix}": v[0].detach().item()
            for k, v in zip(self.keys, self.values)
        }
        return results_dict


class EvalMetrics:
    """Metrics wrapper class for evaluation metrics"""

    metrics_map: dict[str, type[Metric]] = {
        "MSE": MeanSquaredError,
        "R2": R2Score,
        "PSNR": PeakSignalNoiseRatio,
        "SSIM": StructuralSimilarity,
        "FID": FrechetInceptionDistance,
    }

    metrics_kwargs: dict[str, Any] = {
        "multioutput": "uniform_average",
        "num_regressors": 0,
        "data_range": 1.0,
    }
    
    def __init__(
        self,
        metrics: Optional[list[str]] = None,
        device: Optional[torch.device] = None,
        **kwargs: Any
    ):
        """
        Constructor
        """
        self.keys: list[str] = []
        self.values: list[torch.Tensor] = []
        self.fns: list[Metric] = []
        
        # standardize keys from various inputs
        if metrics is None:
            self.keys.append("MSE")
        else:
            for key in self.metrics_map.keys():
                if key in metrics or \
                   key.lower() in metrics or \
                   f"{key}Score" in metrics:
                    self.keys.append(key)
        if not self.keys:
            self.keys.append("MSE")

        # set default keyword arguments for Metric functions
        for k in kwargs:
            if k in self.metrics_kwargs:
                self.metrics_kwargs[k] = kwargs[k]

        self.fns = [
            self.metrics_map[k](device=device, **self.metric_fn_kwargs[k])
            for k in self.keys
        ]

    @property
    def metric_fn_kwargs(self) -> dict[str, dict]:
        """Torch loss function arguments"""
        var_names = {
            k: v.__init__.__code__.co_varnames[1:]
            for k, v in self.metrics_map.items()
        }  # noqa
        return {
            k: {v: self.metrics_kwargs[v] for v in var_names[k] if v in self.metrics_kwargs}
            for k in self.keys
        }

    def index(
        self,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
    ) -> list[str]:
        """Keys index for results"""
        if prefix is None:
            prefix = ""
        if postfix is None:
            postfix = ""
        return [f"{prefix}{k}{postfix}" for k in self.keys]

    def results(
        self,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None
    ) -> dict[str, float]:
        """Result dictionary of computed metrics"""
        if prefix is None:
            prefix = ""
        if postfix is None:
            postfix = ""
        return {f"{prefix}{k}{postfix}": v.detach().item() for k, v in zip(self.keys, self.values)}

    def update(self, input: torch.Tensor, target: torch.Tensor):
        """Update states with the prediction and ground truth values."""
        if len(input.shape) < 4:
            for fn in self.fns:
                fn.update(input.squeeze(), target.squeeze())
        else:
            for k, fn in zip(self.keys, self.fns):
                if k in ["FID"]:
                    fn.update(input, True)
                    fn.update(target, False)
                if k in ["SSIM"]:
                    print(input.shape, target.shape)
                fn.update(input, target)

    def compute(self) -> list[torch.Tensor]:
        """Return the computed metrics"""
        self.values.clear()
        for i, key in enumerate(self.keys):
            self.values.append(self.fns[i].compute())
        return self.values

    def reset(self):
        """Reset the metric state variables to their default values"""
        self.values.clear()
        for fn in self.fns:
            fn.reset()

    def state_dict(self) -> dict[str, dict]:
        """Save metric state variables in state_dict"""
        return {key: fn.state_dict() for key, fn in zip(self.keys, self.fns)}

    def load_state_dict(self, state_dict: dict[str, dict], strict: bool = True):
        """Loads metric state variables from state_dict"""
        for key, fn in zip(self.keys, self.fns):
            fn.load_state_dict(state_dict[key], strict=strict)

    def to(self, device: torch.device, *args, **kwargs):
        """Wrapper for torch.to"""
        for fn in self.fns:
            fn.to(device, *args, **kwargs)

    @property
    def device(self) -> torch.device:
        """The last input device of EvalMetric.to"""
        return self.fns[0].device
