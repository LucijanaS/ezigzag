"""
ezigzag.models module

@author: phdenzel
"""
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
    import torch
    configure_torch_backends(verbose=True)
    dataloader = simpsons_dataloader()
    in_channels = 3
    model = VAE(
        dimensions=2,
        in_channels=in_channels,
        n_channels=32,
        latent_dim=8,
        # block_out_channel_mults=(2, 2),
        block_out_channel_mults=(2, 2, 2),
        # down_block_types=("EncoderDownBlock", "EncoderDownBlock"),
        down_block_types=("EncoderDownBlock", "EncoderDownBlock", "EncoderDownBlock"),
        mid_block_type="EncoderMidBlock",
        # up_block_types=("EncoderUpBlock", "EncoderUpBlock"),
        up_block_types=("EncoderUpBlock", "EncoderUpBlock", "EncoderUpBlock"),
    )
    summary(model, (2, in_channels, 256, 256),
            depth=7,
            col_names=["input_size", "output_size", "num_params"],
            device="cpu")

    recon = model.decode(torch.randn((1, 8, 64, 64)))\
                 .detach()\
                 .cpu()\
                 .squeeze()\
                 .permute(1, 2, 0)\
                 .numpy()
    print(recon.shape)
