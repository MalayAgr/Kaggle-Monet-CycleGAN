import torch
import torch.nn as nn

from .discriminator import Discriminator
from .generator import Generator


class CycleGAN(nn.Module):
    def __init__(
        self,
        monet_gen: Generator,
        monet_disc: Discriminator,
        img_gen: Generator,
        img_disc: Discriminator,
    ) -> None:
        super().__init__()
        self.monet_gen = monet_gen
        self.monet_disc = monet_disc
        self.img_gen = img_gen
        self.img_disc = img_disc

    def forward(
        self, img: torch.Tensor, monet_img: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # Generate the fake Monet painting from original image
        # Generate the fake image from the original Monet
        fake_monet = self.monet_gen(img)
        fake_img = self.img_gen(monet_img)

        # Recreate the Monet from the fake image
        # Recreate the original image from the fake Monet
        cycled_monet = self.monet_gen(fake_img)
        cycled_img = self.img_gen(fake_monet)

        # Generate the identity images from the images
        identity_monet = self.monet_gen(monet_img)
        identity_img = self.img_gen(img)

        # Use discriminators to classifiy the real images
        # Expected output is close to 1 since the images are real
        real_monet_disc = self.monet_disc(monet_img)
        real_img_disc = self.img_disc(img)

        # Use discriminators to classify the fake images
        # Expected output is close to 0 since the images are fake
        fake_monet_disc = self.monet_disc(fake_monet)
        fake_img_disc = self.img_disc(fake_img)

        # Store all the outputs related to the normal image
        img_output = {
            "fake_img": fake_img,
            "cycled_img": cycled_img,
            "identity_img": identity_img,
            "real_img_disc": real_img_disc,
            "fake_img_disc": fake_img_disc,
        }

        # Store all the outputs related to the Monet image
        monet_output = {
            "fake_monet": fake_monet,
            "cycled_monet": cycled_monet,
            "identity_monet": identity_monet,
            "real_monet_disc": real_monet_disc,
            "fake_monet_disc": fake_monet_disc,
        }

        # Return all the outputs
        return {
            **img_output,
            **monet_output,
        }


def get_halved_model(device: torch.device = None) -> CycleGAN:
    monet_gen = Generator(
        conv_out_channels=[32, 64, 128],
        res_out_channels=128,
        transpose_out_channels=[64, 32],
        n_residuals=3,
    )

    monet_disc = Discriminator(out_channels=[32, 64, 128, 256])

    img_gen = Generator(
        conv_out_channels=[32, 64, 128],
        res_out_channels=128,
        transpose_out_channels=[64, 32],
        n_residuals=3,
    )

    img_disc = Discriminator(out_channels=[32, 64, 128, 256])

    cycle_gan = CycleGAN(
        monet_gen=monet_gen,
        monet_disc=monet_disc,
        img_gen=img_gen,
        img_disc=img_disc,
    )

    if device is not None:
        monet_gen = monet_gen.to(device)
        monet_disc = monet_disc.to(device)
        img_gen = img_gen.to(device)
        img_disc = img_disc.to(device)
        cycle_gan = cycle_gan.to(device)

    return cycle_gan
