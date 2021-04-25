from enum import Enum
from typing import Type

import torch
import torch.nn as nn

from cvmodels import ModuleBase


class ResidualBlock(nn.Module):
    """Simple interface implemented by actual building blocks.
    """


class StandardBlock(ResidualBlock):
    """Implementation of the basic building block of a residual network.
    The module is double conv. (3x3, 3x3), followed by the addition of the (possibly scaled) input.
    Expansion is a static property of residual blocks, indicating the channel growth factor in output.
    This is the standard version of the block, without bottleneck.
    """
    expansion = 1

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: nn.Module = None,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        super(StandardBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn1 = batch_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = batch_norm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample or nn.Identity()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass onthe residual block.
        The input is temporarily stored in the 'residual' variable so that it can be used later.
        The downsample is an identity if no layer was passed during initialization.

        :param batch: input tensor of size [batch, channels, height, width]
        :type batch: torch.Tensor
        :return: tensor of size [batch, out_channels *4, out_h, out_w]
        :rtype: torch.Tensor
        """
        identity = batch
        out = self.conv1(batch)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        return self.relu(out)


class BottleneckBlock(ResidualBlock):
    """Implementation of the basic building block of a residual network.
    The module is basically a triple conv. (1x1, 3x3, 1x1), followed by the addition of the (possibly scaled) input.
    Expansion is a static property of residual blocks, indicating the channel growth factor in output.
    NOTE: Since the stride is implemented in the 3x3 convolution, this is actually a ResNet 1.5 according to:
    https://ngc.nvidia.com/catalog/resources/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: nn.Module = None,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        """Implementation of a single block in a Residual Network (ResNet). A residual block is a double convolution,
        recombined with its own input. Biases in convolutions are missing since they are already included in BatchNorms,
        see https://github.com/KaimingHe/deep-residual-networks/issues/10#issuecomment-194037195.
        The '* 4' operation refers to the final block output, which is always 4 time bigger than the initial expansion.

        :param in_channels: number of input channels (2nd dimension in the batch)
        :type in_channels: int
        :param out_channels: number of output channels, similar to input
        :type out_channels: int
        :param stride: shift of the kernel window at each step in pixels, defaults to 1
        :type stride: int, optional
        :param dilation: how much space between kernel weights in pixels, defaults to 1
        :type dilation: int, optional
        :param downsample: optional downsampling module for the input, defaults to None
        :type downsample: nn.Module, optional
        :param batch_norm: class implementing the batch normalization, defaults to nn.BatchNorm2d
        :type batch_norm: Type[nn.Module], optional
        """
        super(BottleneckBlock, self).__init__()
        batch_norm = batch_norm or nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = batch_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               dilation=dilation,
                               padding=dilation,
                               bias=False)
        self.bn2 = batch_norm(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = batch_norm(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample if downsample else nn.Identity()
        self.dilation = dilation
        self.stride = stride

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass onthe residual block.
        The input is temporarily stored in the 'residual' variable so that it can be used later.
        The downsample is an identity if no layer was passed during initialization.

        :param batch: input tensor of size [batch, channels, height, width]
        :type batch: torch.Tensor
        :return: tensor of size [batch, out_channels *4, out_h, out_w]
        :rtype: torch.Tensor
        """
        identity = batch
        x = self.conv1(batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += self.downsample(identity)
        x = self.relu(x)
        return x


class PretrainedWeights(str, Enum):
    RN18 = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
    RN34 = "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    RN50 = "https://download.pytorch.org/models/resnet50-19c8e357.pth"
    RN101 = "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth"
    RN152 = "https://download.pytorch.org/models/resnet152-b121ed2d.pth"


class ResNetVariants(Enum):
    """Descriptor enum for most the common variants of residual nets.
    """
    RN18 = ((2, 2, 2, 2), StandardBlock, PretrainedWeights.RN18)  # no bottleneck
    RN34 = ((3, 4, 6, 3), StandardBlock, PretrainedWeights.RN34)  # no bottleneck
    RN50 = ((3, 4, 6, 3), BottleneckBlock, PretrainedWeights.RN50)
    RN101 = ((3, 4, 23, 3), BottleneckBlock, PretrainedWeights.RN101)
    RN152 = ((3, 8, 36, 3), BottleneckBlock, PretrainedWeights.RN152)


class ResNet(ModuleBase):
    """Implementation of a Deep residual Network, following the original paper https://arxiv.org/abs/1512.03385
    The architecture follows a dynamic organization, where the final network structure can be tuned via parameters.
    """

    def __init__(self,
                 variant: ResNetVariants = ResNetVariants.RN101,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 pretrained: bool = False):
        """Creates a new ResNet model, with the given characteristics.

        :param layers: list of block count for each layer. A layer defines a group of identical residual blocks.
        Every resnet has 4 major groups: in each group the tensor size stays the same.add()
        :type layers:       Tuple[int, int, int, int]
        :param block:       class implementation for the basic building block, defaults to the Bottleneck block
        :type block:        Type[ResNetBlock]
        :param batch_norm:  pointer to the class implementing the batch normalization layer, defaults to BatchNorm2d
        :type batch_norm:   Type[nn.Module]
        :param in_channels: input channels of the image, defaults to 3
        :type in_channels:  int, optional
        :param num_classes: Number of categories, final output dimension, defaults to 1000 (ILSVRC)
        :type num_classes:  enum, since only two options are typically available
        :param pretrained: Whether to load a pretrained checkpoint or start anew
        :type pretrained:  bool
        """
        super(ResNet, self).__init__()
        layers, block, pretrained_url = variant.value
        self.batch_norm = batch_norm or nn.BatchNorm2d
        # input layers
        self.curr_channels = 64
        self.conv1 = nn.Conv2d(in_channels, self.curr_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = batch_norm(self.curr_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # residual layers, out_channels will be mapped to outx4
        self.layer1 = self._residual_group(block, batch_norm, layers[0], 64, stride=1)
        self.layer2 = self._residual_group(block, batch_norm, layers[1], 128, stride=2)
        self.layer3 = self._residual_group(block, batch_norm, layers[2], 256, stride=2)
        self.layer4 = self._residual_group(block, batch_norm, layers[3], 512, stride=2)
        # final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        if pretrained:
            self._from_pretrained(pretrained_url)

    def _residual_group(self,
                        block: Type[ResidualBlock],
                        batch_norm: Type[nn.Module],
                        residual_blocks: int,
                        out_channels: int,
                        stride: int = 1,
                        dilation: int = 1) -> nn.Sequential:
        """Builds a full ResNet layer/group, identified with blocks with the same color in most graphs.
        In some groups the input dimension and the output dimension are kept equal, but in case of stride
        or different channel count an additional downsampling 1x1 convolution is required.

        :param block: class implementation for the basic building block, defaults to the Bottleneck block
        :type block: Type[ResNetBlock]
        :param batch_norm: batch normalization class
        :type batch_norm: Type[nn.Module]
        :param residual_blocks: amount of basic blocks to be added to the current sequential group
        :type residual_blocks: int
        :param out_channels: number of channels in the output. This will actually be scaled by a factor of 4.
        :type out_channels: int
        :param stride: shift amount for the 3x3 convolution to avoid MaxPooling
        :type stride: int
        :param dilation: alternative to stride, not yet implemented at the moment
        :type dilation: int
        :return: a sequential group with N residual blocks
        :rtype: nn.Sequential
        """
        adapter = None
        layers = list()
        scaled_output = out_channels * block.expansion
        # whenever we require a downsampling for the input
        if stride != 1 or self.curr_channels != (scaled_output):
            adapter = nn.Sequential(
                nn.Conv2d(self.curr_channels, scaled_output, kernel_size=1, stride=stride, bias=False),
                batch_norm(scaled_output))
        # append downsampling layer
        layers.append(
            block(self.curr_channels,
                  out_channels,
                  stride=stride,
                  dilation=dilation,
                  downsample=adapter,
                  batch_norm=batch_norm))
        self.curr_channels = scaled_output
        # append the other required layers, starting from 1
        for _ in range(1, residual_blocks):
            layers.append(block(self.curr_channels, out_channels, dilation=dilation, batch_norm=batch_norm))
        return nn.Sequential(*layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Computes a forward pass over the whole ResNet classifier.

        :param batch: tensor input of size [batch, channels, height, width] where channels depend on images
        :type batch: torch.Tensor
        :return: Tensor of size [batch, num_classes] containing the forward pass results
        :rtype: torch.Tensor
        """
        # initial block
        x = self.conv1(batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # final classifier, flatten but not the batch
        x = self.avgpool(x)
        x = self.fc(torch.flatten(x, start_dim=1))
        return x


class ResNet18(ResNet):
    """Wrapper around the main ResNet class, providing default params for ResNet18.
    """

    def __init__(self,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 pretrained: bool = False):
        super().__init__(variant=ResNetVariants.RN18,
                         batch_norm=batch_norm,
                         in_channels=in_channels,
                         num_classes=num_classes,
                         pretrained=pretrained)


class ResNet34(ResNet):
    """Wrapper around the main ResNet class, providing default params for ResNet34.
    """

    def __init__(self,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 pretrained: bool = False):
        super().__init__(variant=ResNetVariants.RN34,
                         batch_norm=batch_norm,
                         in_channels=in_channels,
                         num_classes=num_classes,
                         pretrained=pretrained)


class ResNet50(ResNet):
    """Wrapper around the main ResNet class, providing default params for ResNet50.
    """

    def __init__(self,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 pretrained: bool = False):
        super().__init__(variant=ResNetVariants.RN50,
                         batch_norm=batch_norm,
                         in_channels=in_channels,
                         num_classes=num_classes,
                         pretrained=pretrained)


class ResNet101(ResNet):
    """Wrapper around the main ResNet class, providing default params for ResNet101.
    """

    def __init__(self,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 pretrained: bool = False):
        super().__init__(variant=ResNetVariants.RN101,
                         batch_norm=batch_norm,
                         in_channels=in_channels,
                         num_classes=num_classes,
                         pretrained=pretrained)


class ResNet152(ResNet):
    """Wrapper around the main ResNet class, providing default params for ResNet152.
    """

    def __init__(self,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 pretrained: bool = False):
        super().__init__(variant=ResNetVariants.RN152,
                         batch_norm=batch_norm,
                         in_channels=in_channels,
                         num_classes=num_classes,
                         pretrained=pretrained)


if __name__ == "__main__":
    x = torch.rand((1, 3, 224, 224))
    model = ResNet(pretrained=True)
    print(model)
    print(model(x).size())
