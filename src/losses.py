import torch
import torch.nn as nn


def _discriminator_loss(
    real_disc: torch.Tensor, fake_disc: torch.Tensor
) -> torch.Tensor:
    # Discriminator is a binary classifier and uses BCE
    loss_fn = nn.BCEWithLogitsLoss()

    # The real image's y_true is all ones
    real_loss = loss_fn(real_disc, torch.ones_like(real_disc))
    # The fake image's y_true is all zeros
    fake_loss = loss_fn(fake_disc, torch.zeros_like(fake_disc))

    return real_loss + fake_loss


def cycle_consistency_loss(
    original: torch.Tensor, cycled: torch.Tensor, scale: float = 10.0
) -> torch.Tensor:
    # Use L1 distance as loss function with mean reduction
    loss_fn = nn.L1Loss(reduction="mean")
    return loss_fn(cycled, original) * scale


def identity_loss(
    original: torch.Tensor, identity: torch.Tensor, scale: float = 10.0
) -> torch.Tensor:
    # Use L1 distance as loss function with mean reduction
    loss_fn = nn.L1Loss(reduction="mean")
    return loss_fn(identity, original) * scale * 0.5


def _generator_loss(
    original: torch.Tensor,
    fake_disc: torch.Tensor,
    identity: torch.Tensor,
    total_ccl: torch.Tensor,
    scale: float = 10.0,
) -> torch.Tensor:
    id_loss = identity_loss(original, identity, scale=scale)

    loss_fn = nn.BCEWithLogitsLoss()
    # The goal of the generator is to fool the discriminator into thinking
    # The generated image is real
    # Hence, the loss function all ones as its y_true
    gen_loss = loss_fn(fake_disc, torch.ones_like(fake_disc))

    return total_ccl + id_loss + gen_loss


def discriminator_loss(
    real_img_disc: torch.Tensor,
    fake_img_disc: torch.Tensor,
    real_monet_disc: torch.Tensor,
    fake_monet_disc: torch.Tensor,
) -> torch.Tensor:
    # Discriminator loss for real-life photos
    img_disc_loss = _discriminator_loss(
        real_disc=real_img_disc, fake_disc=fake_img_disc
    )

    # Discriminator loss for Monet paintings
    monet_disc_loss = _discriminator_loss(
        real_disc=real_monet_disc, fake_disc=fake_monet_disc
    )

    # Total loss is the sum of both divided by 2
    return (img_disc_loss + monet_disc_loss) * 0.5


def generator_loss(
    img: torch.Tensor,
    fake_img_disc: torch.Tensor,
    cycled_img: torch.Tensor,
    identity_img: torch.Tensor,
    monet: torch.Tensor,
    fake_monet_disc: torch.Tensor,
    cycled_monet: torch.Tensor,
    identity_monet: torch.Tensor,
    scale: float = 10,
) -> torch.Tensor:

    # Calculate total cycle consistency loss
    img_cc_loss = cycle_consistency_loss(original=img, cycled=cycled_img, scale=scale)
    monet_cc_loss = cycle_consistency_loss(
        original=monet, cycled=cycled_monet, scale=scale
    )
    total_ccl = img_cc_loss + monet_cc_loss

    # Calculate the generator loss for real-life photos
    # It is: total_ccl + identity loss + loss from discriminator
    img_gen_loss = _generator_loss(
        original=img,
        fake_disc=fake_img_disc,
        identity=identity_img,
        total_ccl=total_ccl,
        sclae=scale,
    )

    # Calculate the generator loss Monet paintings
    # It is: total_ccl + identity loss + loss from discriminator
    monet_gen_loss = _generator_loss(
        original=monet,
        fake_disc=fake_monet_disc,
        identity=identity_monet,
        total_ccl=total_ccl,
        scale=scale,
    )

    # Total loss is the sum of both
    return img_gen_loss + monet_gen_loss
