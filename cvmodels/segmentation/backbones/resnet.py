from enum import Enum
from typing import List, Tuple, Type

import torch
import torch.nn as nn

from cvmodels.classification.resnet import ResNet, ResidualBlock, ResNetVariants
from cvmodels.segmentation.backbones import Backbone


class OutputStrides(Enum):
    OS16 = ((1, 2, 2, 1), (1, 1, 1, 2))  # output stride == 16
    OS08 = ((1, 2, 1, 1), (1, 1, 2, 4))  # output stride == 8


class ResNetBackbone(ResNet, Backbone):
    """Implementation of a Deep residual Network, following the original paper https://arxiv.org/abs/1512.03385
    and the modifications introduced by deeplab in order to adapt the atrous convolutions.
    The architecture follows a dynamic organization, where the final network structure can be tuned via parameters.
    """

    def __init__(self,
                 variant: ResNetVariants = ResNetVariants.RN101,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d,
                 output_strides: OutputStrides = OutputStrides.OS16,
                 in_channels: int = 3):
        """Creates a new instance of Residual Network as a segmentation backbone.

        :param variant: Which variant of ResNet to build (sets the basic block), defaults to ResNetVariants.RN101
        :type variant: ResNetVariants, optional
        :param batch_norm: which batch normalization class, defaults to nn.BatchNorm2d
        :type batch_norm: Type[nn.Module], optional
        :param output_strides: [description], defaults to OutputStrides.OS16
        :type output_strides: OutputStrides, optional
        :param in_channels: how many channels for the input images, defaults to 3
        :type in_channels: int, optional
        """
        super(ResNet, self).__init__()
        layers, block = variant.value
        strides, dilations = output_strides.value
        # input layers
        self.output_stride = 16 if output_strides == OutputStrides.OS16 else 8
        self.curr_channels = 64
        self.conv1 = nn.Conv2d(in_channels, self.curr_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = batch_norm(self.curr_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # residual layers, out_channels will be mapped to outx4
        self.layer1 = self.residual_group(block, batch_norm, layers[0], 64, strides[0], dilations[0])
        self.layer2 = self.residual_group(block, batch_norm, layers[1], 128, strides[1], dilations[1])
        self.layer3 = self.residual_group(block, batch_norm, layers[2], 256, strides[2], dilations[2])
        self.layer4 = self.backbone_group(block, batch_norm, 512, stride=strides[3], dilation=dilations[3])

    def scaling_factor(self) -> int:
        return self.output_stride

    def output_features(self) -> int:
        return 2048, 256

    def backbone_group(self,
                       block: Type[ResidualBlock],
                       batch_norm: Type[nn.Module],
                       out_channels: int,
                       residual_blocks: List[int] = [1, 2, 4],
                       stride: int = 1,
                       dilation: int = 1) -> nn.Sequential:
        """Builds the last layer of a ResNet backbone, which requires more dilations,
        In particular, residual blocks does not contain the number of basic blocks to be added,
        but the dilations for the next three customized components.

        :param block: block class to be instantiated
        :type block: Type[ResidualBlock]
        :param batch_norm: batch normalization class to be instantiated
        :type batch_norm: Type[nn.Module]
        :param out_channels: number of output channels
        :type out_channels: int
        :param residual_blocks: list of dilations for the last block, defaults to [1, 2, 4]
        :type residual_blocks: List[int], optional
        :param stride: stride for the convolutions, defaults to 1
        :type stride: int, optional
        :param dilation: dilation for the convolutions, defaults to 1
        :type dilation: int, optional
        :return: sequential module containing the last block
        :rtype: nn.Sequential
        """
        adapter = None
        layers = list()
        scaled_output = out_channels * block.expansion
        # whenever we require a downsampling for the inpu
        if stride != 1 or self.curr_channels != (scaled_output):
            adapter = nn.Sequential(
                nn.Conv2d(self.curr_channels, scaled_output, kernel_size=1, stride=stride, bias=False),
                batch_norm(scaled_output))
        layers.append(
            block(self.curr_channels,
                  out_channels,
                  stride=stride,
                  dilation=residual_blocks[0]*dilation,
                  downsample=adapter,
                  batch_norm=batch_norm))
        self.curr_channels = scaled_output
        # append the other required layers, starting from 1
        for b in residual_blocks:
            layers.append(block(self.curr_channels, out_channels, dilation=b*dilation, batch_norm=batch_norm))
        return nn.Sequential(*layers)

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes a forward pass over the whole ResNet backbone.
        The output is not a class, but a tensor with dimensions [batch, 2048, height/N, width/N], where
        N is the output stride, equal to 16 or 8.

        :param batch: tensor input of size [batch, channels, height, width] where channels depend on images
        :type batch: torch.Tensor
        :return: Tuple containing two tensors of size [batch, channels, h, w], with one representing a lower level info
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        # initial block, same as standard resnet
        x = self.conv1(batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # residual blocks, without classifier head
        x = self.layer1(x)
        skip = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, skip


if __name__ == "__main__":
    """
    Simple test that can be run as module.
    """
    x = torch.rand((1, 3, 512, 512))
    model = ResNetBackbone(variant=ResNetVariants.RN50, output_strides=OutputStrides.OS16)
    model.eval()
    with torch.no_grad():
        a, b = model(x)
        print(a.size(), b.size())
