from collections import OrderedDict
from enum import Enum
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from cvmodels.functional import fixed_padding
from cvmodels.segmentation.backbones import Backbone, LOG


class OutputStrides(Enum):
    """Size of the final output stride, the bigger it is, the smaller the feature maps.
    """
    OS16 = (2, 1, (1, 2))  # output stride == 16
    OS08 = (1, 2, (2, 4))  # output stride == 8


class MiddleFlows(Enum):
    """How many layers in the middle flow, only two options.
    """
    MF16 = 16
    MF08 = 8


class PretrainedWeights(str, Enum):
    MF16 = "https://github.com/edornd/cv-models/releases/download/v0.1-xception/xception-backbone-m16-b5690688.pth"
    MF08 = "https://github.com/edornd/cv-models/releases/download/v0.1-xception/xception-backbone-m08-b5690688.pth"


class XceptionVariants(Enum):
    MF16 = (MiddleFlows.MF16, PretrainedWeights.MF16)
    MF08 = (MiddleFlows.MF08, PretrainedWeights.MF08)


class SeparableConv2d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 bias: bool = False,
                 batch_norm: nn.Module = None):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               in_channels,
                               kernel_size,
                               stride=stride,
                               padding=0,
                               dilation=dilation,
                               groups=in_channels,
                               bias=bias)
        self.bn = batch_norm(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        return self.pointwise(x)


class XceptionBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 repetitions: int,
                 stride: int = 1,
                 dilation: int = 1,
                 batch_norm: nn.Module = None,
                 start_with_relu: bool = True,
                 grow_first: bool = True,
                 is_last: bool = False):
        super(XceptionBlock, self).__init__()
        # add a skip layer when reducing by channels or stride
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                                      batch_norm(out_channels))
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        filters = in_channels
        rep = []
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_channels, out_channels, 3, 1, dilation, batch_norm=batch_norm))
            rep.append(batch_norm(out_channels))
            filters = out_channels

        for _ in range(repetitions - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, batch_norm=batch_norm))
            rep.append(batch_norm(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_channels, out_channels, 3, 1, dilation, batch_norm=batch_norm))
            rep.append(batch_norm(out_channels))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(out_channels, out_channels, 3, 2, batch_norm=batch_norm))
            rep.append(batch_norm(out_channels))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(out_channels, out_channels, 3, 1, batch_norm=batch_norm))
            rep.append(batch_norm(out_channels))

        if not start_with_relu:
            rep = rep[1:]
        self.rep = nn.Sequential(*rep)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x = self.rep(batch)
        skip = self.skip(batch)
        x = x + skip
        return x


class XceptionBackbone(Backbone):
    """Headless Xception model, providing encoded features for DeepLab and other segmentation heads.
    Compared to the classification version, the Xception backbone has:
     - more layers (16 instead of 8 middle flow blocks)
     - no max pooling, replaced with higher strides in the convolutions
     - extra batch norms and ReLUs
    In this case, the number of middle flow layers are customizable.
    Original paper: https://arxiv.org/pdf/1802.02611.pdf
    Implementation following https://github.com/jfzhang95/pytorch-deeplab-xception
    """

    def __init__(self,
                 in_channels: int = 3,
                 output_strides: OutputStrides = OutputStrides.OS08,
                 variant: XceptionVariants = XceptionVariants.MF16,
                 batch_norm: nn.Module = nn.BatchNorm2d,
                 pretrained: bool = True):
        super(XceptionBackbone, self).__init__()
        middle_flow, pretrain_url = variant.value
        # if the number of input channels is not 3, warn the user
        if in_channels != 3 and pretrained:
            LOG.warning("Using a number of input channels != 3 with pretrained model, are you sure?")

        block3_stride, mid_dilation, end_dilation = output_strides.value
        self.output_stride = 16 if output_strides == OutputStrides.OS16 else 8
        # Entry flow
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = batch_norm(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = batch_norm(64)
        # functional relu here in forward
        self.block1 = XceptionBlock(64, 128, 2, stride=2, batch_norm=batch_norm, start_with_relu=False)
        self.block2 = XceptionBlock(128, 256, 2, stride=2, batch_norm=batch_norm, start_with_relu=True, grow_first=True)
        self.block3 = XceptionBlock(256, 728, 2, stride=block3_stride, batch_norm=batch_norm,
                                    start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow, dynamic count of modules
        middle_blocks = []
        for i in range(middle_flow.value):
            name = f"block{i + 4}"
            block = XceptionBlock(728, 728, 3, stride=1, dilation=mid_dilation, batch_norm=batch_norm,
                                  start_with_relu=True, grow_first=True)
            middle_blocks.append((name, block))
        self.mid_flow = nn.Sequential(OrderedDict(middle_blocks))

        # Exit flow
        self.exit_block = XceptionBlock(728, 1024, 2, stride=1, dilation=end_dilation[0], batch_norm=batch_norm,
                                        start_with_relu=True, grow_first=False, is_last=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=end_dilation[1], batch_norm=batch_norm)
        self.bn3 = batch_norm(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=end_dilation[1], batch_norm=batch_norm)
        self.bn4 = batch_norm(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=end_dilation[1], batch_norm=batch_norm)
        self.bn5 = batch_norm(2048)
        if pretrained:
            self._from_pretrained(pretrain_url.value)

    def scaling_factor(self) -> int:
        return self.output_stride

    def output_features(self) -> int:
        return 2048, 128

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Entry flow
        x = self.conv1(batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.block1(x)
        skip = x
        x = F.relu(x)
        x = self.block2(x)
        x = self.block3(x)
        # Middle flow
        x = self.mid_flow(x)
        # Exit flow
        x = self.exit_block(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        return x, skip


if __name__ == "__main__":
    """
    Simple test that can be run as module.
    """
    x = torch.rand((1, 3, 512, 512))
    model = XceptionBackbone(in_channels=3,
                             output_strides=OutputStrides.OS16,
                             variant=XceptionVariants.MF08,
                             pretrained=True)
    model.eval()
    with torch.no_grad():
        a, b = model(x)
        print(a.size(), b.size())
