import torch.nn as nn
import torch


class ConvBlockD(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, *, norm: bool = True
    ) -> None:
        super().__init__()

        # Create Conv layer
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
                padding_mode="reflect",
            )
        ]

        # If norm is required, add the norm layer
        if norm is True:
            layers.append(nn.InstanceNorm2d(out_channels))

        # Add ReLU activation
        # Note: Paper uses LeakyReLU but that could be because
        # Of when it was written
        layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: list[int] = None) -> None:
        super().__init__()

        # If not provided, set out_channels to default value
        if out_channels is None:
            out_channels = [64, 128, 256, 512]

        out_c = out_channels[0]

        # Add first hidden layer which has no InstanceNorm2d
        layers = [
            ConvBlockD(
                in_channels=in_channels, out_channels=out_c, stride=2, norm=False
            )
        ]

        # Add the remaining hidden layers except the last one
        in_c = out_c
        for out_c in out_channels[1:-1]:
            layers.append(ConvBlockD(in_channels=in_c, out_channels=out_c, stride=2))
            in_c = out_c

        # Add final hidden layer which has stride of 1
        layers.append(
            ConvBlockD(in_channels=in_c, out_channels=out_channels[-1], stride=1)
        )

        in_c = out_channels[-1]

        # Add output layer
        # Since the discriminator is a binary classifier, out_channels is 1
        layers.append(
            nn.Conv2d(
                in_channels=in_c,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sigmoid is applied as part of the loss function
        return self.model(x)
