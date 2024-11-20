"""
ezigzag.train module

Main training framework for ezigzag models

@author: phdenzel
"""
import time
import pprint
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from ezigzag import configure_torch_backends, parse_args
from ezigzag.data import simpsons_dataloader
from ezigzag.models import VAE
from ezigzag.metrics import Objective, EvalMetrics
from ezigzag.utils import (
    save_config,
    load_config,
    write_csv_log,
    write_ckpt
)


def train_model(**kwargs):
    """
    Model training method
    """
    print("Configuration:")
    kwargs = parse_args(**kwargs) | kwargs
    verbose = kwargs["verbose"]
    if kwargs.get("config_file", None):
        kwargs |= load_config(kwargs["config_file"])
    save_config(kwargs, kwargs["ckpt_dir"], include_git_state=True)
    pprint.pprint(kwargs)

    device = kwargs["device"]
    configure_torch_backends(verbose=verbose)

    # Dataset
    dataloader = simpsons_dataloader()
    data_shape = dataloader.dataset[0][0].shape
    print("Data shape: ", data_shape)
    # Model
    model = VAE(
        dimensions=2,
        in_channels=3,
        n_channels=32,
        latent_dim=4,
        block_out_channel_mults=(2, 2),
        down_block_types=("EncoderDownBlock", "EncoderDownBlock"),
        mid_block_type="EncoderMidBlock",
        up_block_types=("EncoderUpBlock", "EncoderUpBlock"),
    )
    model.to(device)

    # Optimizer
    optimizer = Adam(model.parameters(), kwargs["learning_rate"])
    scheduler = StepLR(optimizer, kwargs["lr_step"], gamma=kwargs["lr_decay"])

    # Losses and metrics
    objective_fn = Objective(**kwargs)
    metrics = EvalMetrics(**kwargs)
    train_csv = write_csv_log(
        dict.fromkeys(["epoch", "iteration", "lr"] + objective_fn.index("train_")),
        filename=kwargs["results_dir"] / kwargs["train_log"],
        write_row=True,
        write_header=True,
    )
    eval_csv = write_csv_log(
        dict.fromkeys(["epoch"] + metrics.index()),
        filename=kwargs["results_dir"] / kwargs["eval_log"],
        write_row=True,
        write_header=True,
    )

    # Checkpoints
    write_ckpt(kwargs["ckpt_dir"], model, optimizer, scheduler, epoch=-1)

    # Main training loop
    t_ini = time.time()
    iteration_total = 0
    for epoch in range(kwargs["epoch_start"], kwargs["epoch_start"]+kwargs["n_epochs"]):
        t_epoch_ini = time.time()

        # Training
        model.train()
        dataloader.train()
        iterator = dataloader.tqdm(desc="Training")
        for iteration, batch in iterator:
            data = batch[0].to(device)
            dist = model.encode(data)
            z = dist.sample()
            recon = model.decode(z)
            loss = objective_fn(recon, data, index=0)
            loss += objective_fn(dist, index=1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss = {
                "epoch": epoch,
                "iteration": iteration,
                "lr": scheduler.get_last_lr()[0],
            } | objective_fn.results(prefix="train_")
            if (iteration % kwargs["freq_stdout"] == 0 or iteration == iterator.total) and verbose:
                iterator.set_postfix(train_loss)
            if iteration % kwargs["freq_log"] == 0 or iteration == iterator.total:
                write_csv_log(train_loss, _file=train_csv)
            iteration_total += 1
        scheduler.step()

        t_mid = time.time()
        if verbose:
            print(f"Epoch: {epoch:4d} - Training time: {t_mid - t_epoch_ini:4.2f}s")

        # Checkpoint
        if epoch % kwargs["freq_ckpt"] == 0 or epoch == kwargs["epoch_start"]+kwargs["n_epochs"]-1:
            write_ckpt(kwargs["ckpt_dir"], model, optimizer, scheduler, epoch=epoch)
        write_ckpt(kwargs["ckpt_dir"], model, optimizer, scheduler, epoch=epoch, as_latest=True)

        # # Evaluation
        # model.eval()
        # dataloader.eval()
        # with torch.no_grad():
        #     iterator = dataloader.tqdm(desc="Evaluation")
        #     for iteration, batch in iterator:
        #         data = batch[0].to(device)
        #         pred = model(data)
        #         metrics.update(pred, data)
        #     metrics.compute()
        #     val_loss = {"epoch": epoch} | metrics.results()
        #     metrics.reset()
        #     write_csv_log(val_loss, _file=eval_csv)
        #     if verbose:
        #         print("Evaluation: ", "   ".join([f"{k}: {v}" for k, v in val_loss.items()]))
        # t_epoch_fin = time.time()
        # if verbose:
        #     print(f"Epoch: {epoch} - Eval time: {t_epoch_fin - t_mid:4.2f}s")
    t_fin = time.time()
    train_csv.close()
    eval_csv.close()
    if verbose:
        print(f"Total training time: {t_fin - t_ini:4.2f}s")


if __name__ == "__main__":
    config = parse_args()
    # Trigger model training
    config["ckpt_dir"].mkdir(parents=True, exist_ok=True)
    config["results_dir"].mkdir(parents=True, exist_ok=True)
    config["loss"] = ["MSE", "ELBO"]
    config["metrics"] = []
    config["verbose"] = True
    train_model(**config)
