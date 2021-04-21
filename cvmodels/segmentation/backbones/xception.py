from collections import OrderedDict
from enum import Enum
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

from cvmodels.classification.xception import XceptionBlock, SeparableConv2d


PRETRAINED_URL = "http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth"


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


class XceptionBackbone(nn.Module):
    """Headless Xception model, providing encoded features for DeepLab and other segmentation heads.
    Implementation following https://github.com/Cadene/pretrained-models.pytorch and
    https://github.com/mirkozaff/pytorch_segmentation
    """

    def __init__(self, in_channels: int = 3,
                 output_strides: OutputStrides = OutputStrides.OS16,
                 middle_flow: MiddleFlows = MiddleFlows.MF16,
                 batch_norm: nn.Module = nn.BatchNorm2d,
                 weight_init: bool = False,
                 pretrained: bool = True):
        super(XceptionBackbone, self).__init__()
        # Stride for block 3 (entry flow), and the dilation rates for middle flow and exit flow
        stride_block3, dil_mid, dil_ext = output_strides.value
        # Entry Flow
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, padding=1, bias=False)
        self.bn1 = batch_norm(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1, bias=False)
        self.bn2 = batch_norm(64)
        self.block1 = XceptionBlock(64, 128, stride=2, dilation=1, use_first_relu=False, batch_norm=batch_norm)
        self.block2 = XceptionBlock(128, 256, stride=2, dilation=1, batch_norm=batch_norm)
        self.block3 = XceptionBlock(256, 728, stride=stride_block3, dilation=1, batch_norm=batch_norm)
        # Middle Flow
        middle_blocks = []
        for i in range(middle_flow.value):
            block = XceptionBlock(728, 728, stride=1, dilation=dil_mid, batch_norm=batch_norm)
            middle_blocks.append((f"block{i + 4}", block))
        self.mid_flow = nn.Sequential(OrderedDict(middle_blocks))
        # Exit flow
        self.exit_block = XceptionBlock(728, 1024, stride=1, dilation=dil_ext[0], exit_flow=True, batch_norm=batch_norm)
        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=dil_ext[1])
        self.bn3 = batch_norm(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=dil_ext[1])
        self.bn4 = batch_norm(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=dil_ext[1])
        self.bn5 = batch_norm(2048)

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor]:
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
                             middle_flow=MiddleFlows.MF08,
                             pretrained=False)
    model.eval()
    print(model)
    with torch.no_grad():
        a, b = model(x)
        print(a.size(), b.size())
