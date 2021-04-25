from typing import Optional

import torch
import torch.nn as nn


def convolution_block(in_channels: int, out_channels: int, mid_channels: Optional[int] = None) -> nn.Module:
    """Creates a sequential block, containing two composite layers in series with
    (convolution, batch norm, relu activation).

    :param in_channels: number of input channels for the current block (2nd dimension of the batch)
    :type in_channels:  int
    :param out_channels: number of output channels for the current block
    :type out_channels: int
    :param mid_channels: number of output channels for the middle block
    :type mid_channels: int
    :return:            Sequential module containing a double 3x3 convolution with BN and ReLU activations
    :rtype:             nn.Module
    """
    mid_channels = mid_channels or out_channels
    return nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                         nn.BatchNorm2d(num_features=mid_channels),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                         nn.BatchNorm2d(num_features=out_channels),
                         nn.ReLU(inplace=True))


class EncoderBlock(nn.Module):
    """Basic block to define the encoding basic block with pooling and double convolution.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = convolution_block(in_channels=in_channels, out_channels=out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, batch: torch.Tensor):
        x = self.max_pool(batch)
        return self.conv_block(x)


class DecoderBlock(nn.Module):
    """Basic block describing the upsampling and double convolution procedure.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                          nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.conv_block = convolution_block(in_channels=in_channels, out_channels=out_channels)
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv_block = convolution_block(in_channels, out_channels)

    def forward(self, batch: torch.Tensor, skip: torch.Tensor):
        upsampled = self.upsample(batch)
        return self.conv_block(torch.cat((skip, upsampled), dim=1))


class UNet(nn.Module):
    """Implementation of the UNet model, a U-shaped encoder-decoder architecture for semantic segmentation.
    The structure remains as close as possible to the paper (https://arxiv.org/abs/1505.04597), with the exception
    of the extra padding in the convolutional blocks, which helps maintaining the output shape equal to the input,
    instead of the crops proposed in the paper.
    The bilinear upsampling is also provided as option: while not being strictly compliant, it should provide better
    results without "checkboards" artifacts (https://distill.pub/2016/deconv-checkerboard/)
    """

    def __init__(self,
                 input_channels: int = 3,
                 init_features: int = 64,
                 outputs: int = 1,
                 bilinear: bool = True) -> None:
        """Creates a new UNet model, with the given number of input channels, initial features and final output layers.

        :param input_channels: channel count in the input images, defaults to 3 for RGB
        :type input_channels: int, optional
        :param init_features: initial amount of feature maps, defaults to 64
        :type init_features: int, optional
        :param outputs: number of output classes, defaults to 1
        :type outputs: int, optional
        """
        super().__init__()
        f = init_features
        # encoder
        self.input = convolution_block(input_channels, f)
        self.encoder1 = EncoderBlock(f, f * 2)
        self.encoder2 = EncoderBlock(f * 2, f * 4)
        self.encoder3 = EncoderBlock(f * 4, f * 8)
        self.encoder4 = EncoderBlock(f * 8, f * 16)
        self.decoder1 = DecoderBlock(f * 16, f * 8, bilinear=bilinear)
        self.decoder2 = DecoderBlock(f * 8, f * 4, bilinear=bilinear)
        self.decoder3 = DecoderBlock(f * 4, f * 2, bilinear=bilinear)
        self.decoder4 = DecoderBlock(f * 2, f, bilinear=bilinear)
        self.output = nn.Conv2d(f, outputs, kernel_size=1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Encodes the given batch of images into a semantically meaningful vector.

        :param batch:   image batch of size [batch, channels, height, width]
        :type batch:    torch.Tensor
        :return:        tensor [batch, 1024, h, w], where h, w depend on the input dimension
        :rtype:         torch.Tensor
        """
        x1 = self.input(batch)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)
        x = self.decoder1(x5, x4)
        x = self.decoder2(x, x3)
        x = self.decoder3(x, x2)
        x = self.decoder4(x, x1)
        return self.output(x)
