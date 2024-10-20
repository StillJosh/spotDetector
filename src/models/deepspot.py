# deepspot.py
# Description: DeepSpot model for image segmentation. From https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10951802/
# Author: Joshua Stiller
# Date: 16.10.24

from typing import List

import torch
import torch.nn as nn


class IdentityBlock(nn.Module):
    """
    Identity block for the ResNet architecture.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    filters : List[int]
        List of filter sizes [F1, F2, F3].
    kernel_size : int, optional
        Kernel size for the middle convolutional layer. Default is 3.
    dropout_rate : float, optional
        Dropout rate. Default is 0.3.
    """

    def __init__(
            self,
            in_channels: int,
            filters: List[int],
            kernel_size: int = 3,
            dropout_rate: float = 0.3,
    ):
        super(IdentityBlock, self).__init__()
        f1, f2, f3 = filters

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, stride=1, padding=0)

        self.bn2 = nn.BatchNorm2d(f1)
        self.conv2 = nn.Conv2d(
            f1, f2, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

        self.bn3 = nn.BatchNorm2d(f2)
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1, stride=1, padding=0)

        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = self.dropout(out)

        out += shortcut
        return out


class ConvBlock(nn.Module):
    """
    Convolutional block with a shortcut path.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    filters : List[int]
        List of filter sizes [F1, F2, F3].
    kernel_size : int, optional
        Kernel size for the middle convolutional layer. Default is 3.
    stride : int, optional
        Stride for the first convolutional layer. Default is 2.
    dropout_rate : float, optional
        Dropout rate. Default is 0.1.
    """

    def __init__(
            self,
            in_channels: int,
            filters: List[int],
            kernel_size: int = 3,
            stride: int = 2,
            dropout_rate: float = 0.1,
    ):
        super(ConvBlock, self).__init__()
        f1, f2, f3 = filters

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels, f1, kernel_size=1, stride=stride, padding=0
        )

        self.bn2 = nn.BatchNorm2d(f1)
        self.conv2 = nn.Conv2d(
            f1, f2, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

        self.bn3 = nn.BatchNorm2d(f2)
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1, stride=1, padding=0)

        self.dropout = nn.Dropout2d(dropout_rate)

        # Shortcut path
        self.shortcut_conv = nn.Conv2d(
            in_channels, f3, kernel_size=1, stride=stride, padding=0
        )
        self.shortcut_bn = nn.BatchNorm2d(f3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut_conv(x)
        shortcut = self.shortcut_bn(shortcut)

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = self.dropout(out)

        out += shortcut
        return out


class ConvUpBlock(nn.Module):
    """
    Transposed convolutional block for upsampling with a shortcut path.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    filters : List[int]
        List of filter sizes [F1, F2, F3].
    stride : int, optional
        Stride for the transposed convolutional layer. Default is 2.
    dropout_rate : float, optional
        Dropout rate. Default is 0.1.
    """

    def __init__(
            self,
            in_channels: int,
            filters: List[int],
            stride: int = 2,
            dropout_rate: float = 0.1,
    ):
        super(ConvUpBlock, self).__init__()
        f1, f2, f3 = filters

        self.conv_transpose1 = nn.ConvTranspose2d(
            in_channels,
            f1,
            kernel_size=3,
            stride=stride,
            padding=1,
            output_padding=stride - 1,
        )
        self.bn1 = nn.BatchNorm2d(f1)
        self.relu = nn.ReLU(inplace=True)

        self.conv_transpose2 = nn.ConvTranspose2d(
            f1, f2, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(f2)

        self.conv_transpose3 = nn.ConvTranspose2d(
            f2, f3, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(f3)

        self.dropout = nn.Dropout2d(dropout_rate)

        # Shortcut path
        self.shortcut_conv_transpose = nn.ConvTranspose2d(
            in_channels,
            f2,
            kernel_size=3,
            stride=stride,
            padding=1,
            output_padding=stride - 1,
        )
        self.shortcut_bn = nn.BatchNorm2d(f2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut_conv_transpose(x)
        shortcut = self.shortcut_bn(shortcut)

        out = self.conv_transpose1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv_transpose2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv_transpose3(out)
        out = self.bn3(out)

        out = self.dropout(out)

        out += shortcut
        out = self.relu(out)

        return out


class DeepSpotNet(nn.Module):
    """
    DeepSpot model for image segmentation.

    Parameters
    ----------
    input_channels : int, optional
        Number of input channels. Default is 1.
    dropout_rate : float, optional
        Dropout rate. Default is 0.2.
    conv_block4_filters : int, optional
        Number of filters for the conv block. Default is 128.
    identity_block_filters : int, optional
        Number of filters for the identity blocks. Default is 128.
    """

    def __init__(
            self,
            input_channels: int = 1,
            dropout_rate: float = 0.2,
            conv_block4_filters: int = 128,
            identity_block_filters: int = 128,
    ):
        super(DeepSpotNet, self).__init__()
        # Initial Convolution Branch
        self.conv_branch = nn.Sequential(
            nn.Conv2d(
                input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # MaxPool Branch
        self.maxpool_branch = nn.Sequential(
            nn.Conv2d(
                input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Atrous (Dilated) Convolution Branch
        self.atrous_branch = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Concatenated Output Channels after concatenation
        concat_channels = 128 * 3  # Since we have three branches, each ending with 128 channels

        # First ConvBlock
        self.conv_block = ConvBlock(
            in_channels=concat_channels,
            filters=[conv_block4_filters, conv_block4_filters, identity_block_filters],
            kernel_size=3,
            stride=1,
            dropout_rate=dropout_rate,
        )

        # Identity Blocks
        num_identity_blocks = 12  # As per the original code
        identity_blocks_list = []
        for _ in range(num_identity_blocks):
            identity_blocks_list.append(
                IdentityBlock(
                    in_channels=identity_block_filters,
                    filters=[
                        identity_block_filters,
                        identity_block_filters,
                        identity_block_filters,
                    ],
                    kernel_size=3,
                    dropout_rate=dropout_rate,
                )
            )
        self.identity_blocks = nn.Sequential(*identity_blocks_list)

        # Up Convolution Blocks
        self.conv_up_block1 = ConvUpBlock(
            in_channels=identity_block_filters,
            filters=[256, 128, 128],
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv_up_block2 = ConvUpBlock(
            in_channels=128,
            filters=[128, 64, 64],
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.conv_up_block3 = ConvUpBlock(
            in_channels=64,
            filters=[64, 32, 32],
            stride=2,
            dropout_rate=dropout_rate,
        )

        # Output layer
        self.output_layer = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv = self.conv_branch(x)
        x_maxpool = self.maxpool_branch(x)
        x_atrous = self.atrous_branch(x)

        x = torch.cat([x_conv, x_maxpool, x_atrous], dim=1)  # Concatenate along the channel dimension

        x = self.conv_block(x)

        x = self.identity_blocks(x)

        x = self.conv_up_block1(x)
        x = self.conv_up_block2(x)
        x = self.conv_up_block3(x)

        x = self.output_layer(x)
        x = self.sigmoid(x)

        return x
