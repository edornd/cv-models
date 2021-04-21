import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List


class SeparableConv2d(nn.Module):
    """TODO
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 bias: bool = False):
        """TODO

        :param in_channels: input channels, usually 3 (RGB)
        :type in_channels: int
        :param out_channels: output channels for the convolution
        :type out_channels: int
        :param kernel_size: size of the convolutional filter, defaults to 3
        :type kernel_size: int, optional
        :param stride: stride for the convolution, defaults to 1
        :type stride: int, optional
        :param dilation: dilation of the convolution, defaults to 1
        :type dilation: int, optional
        :param bias: whether to include the bias or not, defaults to False
        :type bias: bool, optional
        """
        super(SeparableConv2d, self).__init__()
        padding = dilation if dilation > kernel_size // 2 else kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels,
                               in_channels,
                               kernel_size,
                               stride,
                               padding=padding,
                               dilation=dilation,
                               groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, bias=bias)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x = self.conv1(batch)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    """Basic block for the Xception backbone. TODO
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 dilation: int = 1,
                 exit_flow: bool = False,
                 use_first_relu: bool = True,
                 batch_norm: nn.Module = nn.BatchNorm2d):
        """TODO

        :param in_channels: input channel count
        :type in_channels: int
        :param out_channels: number of output channels for the current block
        :type out_channels: int
        :param stride: stride for the block, defaults to 1
        :type stride: int, optional
        :param dilation: dilation for the block, defaults to 1
        :type dilation: int, optional
        :param exit_flow: whether to include the exit flow, defaults to False
        :type exit_flow: bool, optional
        :param use_first_relu: whether to use the first ReLU, defaults to True
        :type use_first_relu: bool, optional
        :param batch_norm: batch normalization module, defaults to nn.BatchNorm2d
        :type batch_norm: nn.Module, optional
        """
        super(XceptionBlock, self).__init__()

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                                      batch_norm(out_channels))
        else:
            self.skip = nn.Identity()
        repetitions = self._conv_block(in_channels, out_channels, 3, stride=1,
                                       dilation=dilation, batch_norm=batch_norm)
        repetitions.extend(self._conv_block(out_channels, out_channels, 3, stride=1,
                                            dilation=dilation, batch_norm=batch_norm))
        repetitions.extend(self._conv_block(out_channels, out_channels, 3, stride=stride,
                                            dilation=dilation, batch_norm=batch_norm))
        if exit_flow:
            repetitions[3:6] = repetitions[:3]
            repetitions[:3] = self._conv_block(in_channels,
                                               in_channels,
                                               3,
                                               stride=1,
                                               dilation=dilation,
                                               batch_norm=batch_norm)
        if not use_first_relu:
            repetitions = repetitions[1:]
        self.rep = nn.Sequential(*repetitions)

    def _conv_block(self,
                    in_channels: int,
                    out_channels: int,
                    kernel: int,
                    stride: int,
                    dilation: int,
                    bias: bool = False,
                    batch_norm: nn.Module = nn.BatchNorm2d) -> List[nn.Module]:
        """Builds a convolutional block (relu + conv2d + bn) using the given parameters.

        :param in_channels: input channel count
        :type in_channels: int
        :param out_channels: output channel count
        :type out_channels: int
        :param kernel: kernel size for the conv. layer
        :type kernel: int
        :param stride: stride for the conv. layer
        :type stride: int
        :param dilation: dilation of the conv. layer
        :type dilation: int
        :param bias: whether to include a bias or not, defaults to False
        :type bias: bool, optional
        :param batch_norm: batch normalization module, defaults to nn.BatchNorm2d
        :type batch_norm: nn.Module, optional
        :return: list composed of (ReLU, Conv2d, BatchNorm)
        :rtype: List[nn.Module]
        """
        return [
            nn.ReLU(inplace=True),
            SeparableConv2d(in_channels, out_channels, kernel, stride=stride, dilation=dilation, bias=bias),
            batch_norm(out_channels)
        ]

    def forward(self, x):
        output = self.rep(x)
        skip = self.skip(x)
        return output + skip


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1000, batch_norm: nn.Module = nn.BatchNorm2d):
        """Constructs a new XCeption model, with the given output and input.

        :param in_channels: number of input channels, defaults to 3 (RGB)
        :type in_channels: int, optional
        :param num_classes: number of output classes, defaults to 1000 (ImageNet)
        :type num_classes: int, optional
        :param batch_norm: batch normalization layer, defaults to nn.BatchNorm2d
        :type batch_norm: nn.module, optional
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False)
        self.bn1 = batch_norm(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = batch_norm(64)
        self.relu2 = nn.ReLU(inplace=True)
        # do relu here
        self.block1 = XceptionBlock(64, 128, 2, 2, use_first_relu=False, batch_norm=batch_norm)
        self.block2 = XceptionBlock(128, 256, 2, 2, use_first_relu=True, batch_norm=batch_norm)
        self.block3 = XceptionBlock(256, 728, 2, 2, use_first_relu=True, batch_norm=batch_norm)

        self.block4 = XceptionBlock(728, 728, 3, 1, use_first_relu=True, batch_norm=batch_norm)
        self.block5 = XceptionBlock(728, 728, 3, 1, use_first_relu=True, batch_norm=batch_norm)
        self.block6 = XceptionBlock(728, 728, 3, 1, use_first_relu=True, batch_norm=batch_norm)
        self.block7 = XceptionBlock(728, 728, 3, 1, use_first_relu=True, batch_norm=batch_norm)
        self.block8 = XceptionBlock(728, 728, 3, 1, use_first_relu=True, batch_norm=batch_norm)
        self.block9 = XceptionBlock(728, 728, 3, 1, use_first_relu=True, batch_norm=batch_norm)
        self.block10 = XceptionBlock(728, 728, 3, 1, use_first_relu=True, batch_norm=batch_norm)
        self.block11 = XceptionBlock(728, 728, 3, 1, use_first_relu=True, batch_norm=batch_norm)
        self.block12 = XceptionBlock(728, 1024, 2, 2, use_first_relu=True, exit_flow=True, batch_norm=batch_norm)
        # final blocks
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = batch_norm(1536)
        self.relu3 = nn.ReLU(inplace=True)
        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = batch_norm(2048)
        self.fc = nn.Linear(2048, num_classes)

    def features(self, batch: torch.Tensor) -> torch.Tensor:
        x = self.conv1(batch)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        x = nn.ReLU(inplace=True)(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x = self.features(batch)
        x = self.logits(x)
        return x


if __name__ == "__main__":
    """
    Simple test that can be run as module.
    """
    x = torch.rand((1, 3, 512, 512))
    with torch.no_grad():
        model = Xception(in_channels=3, num_classes=10)
        model.eval()
        print(model(x).size())
