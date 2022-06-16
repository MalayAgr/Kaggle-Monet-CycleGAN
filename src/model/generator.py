import torch
import torch.nn as nn


class ConvBlockG(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        transpose: bool = False,
        use_activation: bool = True,
        norm: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        # Select either a ConvTranspose2d or a Conv2d depending on transpose
        klass = nn.ConvTranspose2d if transpose is True else nn.Conv2d

        # Initialize the class with the appropriate arguments
        layers = [klass(in_channels=in_channels, out_channels=out_channels, **kwargs)]

        # If norm is required, add the norm layer
        if norm is True:
            layers.append(nn.InstanceNorm2d(out_channels))

        # If activation is required, add the activation layer
        # Note: This is the same as the paper
        if use_activation is True:
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        # Each residual block is made up of two conv blocks
        # The first one has an activation
        # While the second one does not
        self.block = nn.Sequential(
            ConvBlockG(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            ConvBlockG(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
                use_activation=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip connection
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        conv_out_channels: list[int] = None,
        res_out_channels: int = 256,
        transpose_out_channels: list[int] = None,
        n_residuals: int = 6,
    ):
        super().__init__()

        # If not provided, set conv_out_channels to the default value
        if conv_out_channels is None:
            conv_out_channels = [64, 128, res_out_channels]
        # If provided, make sure that the last value is same as res_out_channels
        elif conv_out_channels[-1] != res_out_channels:
            msg = (
                f"Make sure that the last value (={conv_out_channels[-1]}) "
                "in conv_out_channels is the same as "
                f"res_out_channels (={res_out_channels})"
            )
            raise ValueError(msg)

        # If not provided, set transpose_out_channels to default value
        if transpose_out_channels is None:
            transpose_out_channels = [128, 64]

        out_c = conv_out_channels[0]

        # Add first hidden layer which has no Instance Norm
        layers = [
            ConvBlockG(
                in_channels=in_channels,
                out_channels=out_c,
                kernel_size=7,
                padding=3,
                padding_mode="reflect",
                norm=False,
            )
        ]

        # Add the remaininig Conv layers
        in_c = out_c
        for out_c in conv_out_channels[1:]:
            layers.append(
                ConvBlockG(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )

            in_c = out_c

        # Add all the residual blocks
        layers.extend(
            ResidualBlock(channels=res_out_channels) for _ in range(n_residuals)
        )

        # Add the transpose Conv layers
        in_c = res_out_channels
        for out_c in transpose_out_channels:
            layers.append(
                ConvBlockG(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    transpose=True,
                )
            )
            in_c = out_c

        # Add the output layer
        # The out_channels should be same as the initial in_channels
        layers.append(
            ConvBlockG(
                in_channels=in_c,
                out_channels=in_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tanh puts the values in the range -1 and 1
        # Which is similar to a standardized image
        return torch.tanh(self.model(x))
