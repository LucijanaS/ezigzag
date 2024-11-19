"""
ezigzag.models module

@author: phdenzel
"""
from torch import nn
from torchinfo import summary
from chuchichaestli.models.vae import VAE
from ezigzag import configure_torch_backends
from ezigzag.data import simpsons_dataloader


def forward(self, x):
    dist = self.encode(x)
    z = dist.sample()
    x_tilde = self.decode(z)
    return x_tilde

VAE.forward = forward
    


if __name__ == "__main__":
    configure_torch_backends(verbose=True)
    dataloader = simpsons_dataloader()
    in_channels = 1
    model = VAE(
        dimensions=2,
        in_channels=in_channels,
        n_channels=16,
        latent_dim=16,
        block_out_channel_mults=(2, 2),
        down_block_types=("EncoderDownBlock", "EncoderDownBlock"),
        mid_block_type="EncoderMidBlock",
        up_block_types=("EncoderUpBlock", "EncoderUpBlock"),
        
    )
    summary(model, (1, in_channels, 512, 512),
            depth=7,
            col_names=["input_size", "output_size", "num_params"],
            device="cpu")
