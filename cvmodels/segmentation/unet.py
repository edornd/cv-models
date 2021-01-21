import torch
import torch.nn as nn


class UNet(nn.Module):
    """Implementation of the UNet model, a U-shaped encoder-decoder architecture for semantic segmentation.
    The structure remains as close as possible to the paper (https://arxiv.org/abs/1505.04597), with the exception
    of the extra padding in the convolutional blocks, which helps maintaining the output shape equal to the input,
    instead of the crops proposed in the paper.
    """

    def __init__(self, input_channels: int = 3, init_features: int = 64, outputs: int = 1) -> None:
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
        self.max_pool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1 = self._conv_block(input_channels, f)
        self.encoder2 = self._conv_block(f, f * 2)
        self.encoder3 = self._conv_block(f * 2, f * 4)
        self.encoder4 = self._conv_block(f * 4, f * 8)
        self.encoder5 = self._conv_block(f * 8, f * 16)
        # decoder
        self.upscale1 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.decoder1 = self._conv_block(f * 16, f * 8)
        self.upscale2 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.decoder2 = self._conv_block(f * 8, f * 4)
        self.upscale3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.decoder3 = self._conv_block(f * 4, f * 2)
        self.upscale4 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.decoder4 = self._conv_block(f * 2, f)
        self.output = nn.Conv2d(f, outputs, kernel_size=1)

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Creates a sequential block, containing two composite layers in series with
        (convolution, batch norm, relu activation).

        :param in_channels: number of input channels for the current block (2nd dimension of the batch)
        :type in_channels:  int
        :param out_channels: number of output channels for the current block
        :type out_channels: int
        :return:            Sequential module containing a double 3x3 convolution with BN and ReLU activations
        :rtype:             nn.Module
        """
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                             nn.BatchNorm2d(num_features=out_channels), nn.ReLU(inplace=True),
                             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                             nn.BatchNorm2d(num_features=out_channels), nn.ReLU(inplace=True))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Encodes the given batch of images into a semantically meaningful vector.

        :param batch:   image batch of size [batch, channels, height, width]
        :type batch:    torch.Tensor
        :return:        tensor [batch, 1024, h, w], where h, w depend on the input dimension
        :rtype:         torch.Tensor
        """
        print(batch.size())
        f1 = self.encoder1(batch)
        f2 = self.max_pool2x2(f1)
        f2 = self.encoder2(f2)
        f3 = self.max_pool2x2(f2)
        f3 = self.encoder3(f3)
        f4 = self.max_pool2x2(f3)
        f4 = self.encoder4(f4)
        f5 = self.max_pool2x2(f4)
        f5 = self.encoder5(f5)
        # decoder
        out = self.upscale1(f5)
        out = self.decoder1(torch.cat((out, f4), dim=1))
        out = self.upscale2(out)
        out = self.decoder2(torch.cat((out, f3), dim=1))
        out = self.upscale3(out)
        out = self.decoder3(torch.cat((out, f2), dim=1))
        out = self.upscale4(out)
        out = self.decoder4(torch.cat((out, f1), dim=1))
        return self.output(out)
