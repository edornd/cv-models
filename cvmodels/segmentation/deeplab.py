from enum import Enum
from typing import Type, Tuple
import torch
import torch.nn as nn

from cvmodels.segmentation.backbones import resnet as rn, xception as xc


class ASPPVariants(Enum):
    """Enum describing the possible dilations in the Atrous spatial Pyramid Pooling block.
    There are essentially two combinations, with output stride= 16 (smaller) or 8 (wider)
    """
    OS16 = (1, 6, 12, 18)
    OS08 = (1, 12, 24, 36)


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling module: this block is responsible for the multi-scale feature extraction,
    using multiple parallel convolutional blocks (conv, bn, relu) with different dilations.
    The four feature groups are then recombined into a single tensor together with an upscaled average pooling
    (that contrasts information loss), then again processed by a 1x1 convolution + dropout
    """

    def __init__(self,
                 in_tensor: Tuple[int, int, int] = (2048, 32, 32),
                 variant: ASPPVariants = ASPPVariants.OS16,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        """Creates a new Atrous spatial Pyramid Pooling block. This module is responsible
        for the extraction of features at different scales from the input tensor (which is
        an encoder version of the image with high depth and low height/width).
        The module combines these multi-scale features into a single tensor via 1x convolutions

        :param in_tensor: input dimensions in (channels, height, width), defaults to (2048, 32, 32)
        :type in_tensor: Tuple[int, int, int], optional
        :param variant: which output stride are we dealing with, defaults to ASSPVariants.OS16
        :type variant: ASSPVariants, optional
        :param batch_norm: batch normalization clas to instatiate, defaults to nn.BatchNorm2d
        :type batch_norm: Type[nn.Module], optional
        """
        super().__init__()
        dilations = variant.value
        in_channels, h, w = in_tensor
        self.aspp1 = self.assp_block(in_channels, 256, 1, 0, dilations[0], batch_norm=batch_norm)
        self.aspp2 = self.assp_block(in_channels, 256, 3, dilations[1], dilations[1], batch_norm=batch_norm)
        self.aspp3 = self.assp_block(in_channels, 256, 3, dilations[2], dilations[2], batch_norm=batch_norm)
        self.aspp4 = self.assp_block(in_channels, 256, 3, dilations[3], dilations[3], batch_norm=batch_norm)
        # this is redoncolous, but it's described in the paper: bring it down to 1x1 tensor and upscale
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
                                     batch_norm(256),
                                     nn.ReLU(inplace=True),
                                     nn.Upsample((h, w), mode="bilinear", align_corners=True))
        self.merge = self.assp_block(256 * 5, 256, kernel=1, padding=0, dilation=1, batch_norm=batch_norm)
        self.dropout = nn.Dropout(p=0.5)

    def assp_block(self, in_channels: int, out_channels: int, kernel: int, padding: int, dilation: int,
                   batch_norm: Type[nn.Module]) -> nn.Sequential:
        """Creates a basic ASPP block, a sequential module with convolution, batch normalization and relu activation.

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels (usually fixed to 256)
        :type out_channels: int
        :param kernel: kernel size for the convolution (usually 3)
        :type kernel: int
        :param padding: convolution padding, usually equal to the dilation, unless no dilation is applied
        :type padding: int
        :param dilation: dilation for the atrous convolution, depends on ASPPVariant
        :type dilation: int
        :param batch_norm: batch normalization class yet to be instantiated
        :type batch_norm: Type[nn.Module]
        :return: sequential block representing an ASPP component
        :rtype: nn.Sequential
        """
        module = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel,
                      stride=1,
                      padding=padding,
                      dilation=dilation,
                      bias=False),
            batch_norm(out_channels),
            nn.ReLU(inplace=True))
        return module

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Computes a forward pass on the ASPP module.
        The same input is processed five times with different dilations. Output sizes are the same,
        except for the pooled layer, which requires an upscaling.

        :param batch: input tensor with dimensions [batch, channels, height, width]
        :type batch: torch.Tensor
        :return: output tensor with dimensions [batch, 256, height, width]
        :rtype: torch.Tensor
        """
        x1 = self.aspp1(batch)
        x2 = self.aspp2(batch)
        x3 = self.aspp3(batch)
        x4 = self.aspp4(batch)
        x5 = self.avgpool(batch)
        x5 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.merge(x5)
        return self.dropout(x)


class DecoderV3(nn.Sequential):
    """Decoder for DeepLabV3, consisting of a double convolution and a direct 16X upsampling.
    This is clearly not the best for performance, but, if memory is a problem, this can save a little space.
    """

    def __init__(self,
                 output_stride: int = 16,
                 output_channels: int = 1,
                 dropout: float = 0.1,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        """Decoder output for the simpler DeepLabV3: this module simply processes the ASPP output
        and upscales it to the input size.The 3x3 convolution and the dropout do not appear in the paper,
        but they are implemented in the official release.

        :param output_stride: scaling factor of the backbone, defaults to 16
        :type output_stride: int, optional
        :param output_channels: number of classes in the output mask, defaults to 1
        :type output_channels: int, optional
        :param dropout: dropout probability before the final convolution, defaults to 0.1
        :type dropout: float, optional
        :param batch_norm: batch normalization class, defaults to nn.BatchNorm2d
        :type batch_norm: Type[nn.Module], optional
        """
        super(DecoderV3, self).__init__(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            batch_norm(256),
            nn.ReLU(inplace=True), nn.Dropout(p=dropout),
            nn.Conv2d(256, output_channels, kernel_size=1),
            nn.Upsample(scale_factor=output_stride, mode="bilinear", align_corners=True))


class DecoderV3Plus(nn.Module):
    """DeepLabV3+ decoder branch, with a skip branch embedding low level
    features (higher resolution) into the highly dimensional output. This typically
    produces much better results than a naive 16x upsampling.
    Original paper: https://arxiv.org/abs/1802.02611
    """

    def __init__(self,
                 low_level_channels: int,
                 output_stride: int = 16,
                 output_channels: int = 1,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        """Returns a new Decoder for DeepLabV3+.
        The upsampling is divided into two parts: a fixed 4x from 128 to 512, and a 2x or 4x
        from 32 or 64 (when input=512x512) to 128, depending on the output stride.

        :param low_level_channels: how many channels on the lo-level skip branch
        :type low_level_channels: int
        :param output_stride: downscaling factor of the backbone, defaults to 16
        :type output_stride: int, optional
        :param output_channels: how many outputs, defaults to 1
        :type output_channels: int, optional
        :param batch_norm: batch normalization module, defaults to nn.BatchNorm2d
        :type batch_norm: Type[nn.Module], optional
        """
        super().__init__()
        low_up_factor = 4
        high_up_factor = output_stride / low_up_factor
        self.low_level = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            batch_norm(48),
            nn.ReLU(inplace=True))
        self.upsample = nn.Upsample(scale_factor=high_up_factor, mode="bilinear", align_corners=True)

        # Table 2, best performance with two 3x3 convs
        self.output = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1, bias=False),
            batch_norm(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            batch_norm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, output_channels, 1, stride=1),
            nn.Upsample(scale_factor=low_up_factor, mode="bilinear", align_corners=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass on the decoder. Low-level features 'skip' are processed and merged
        with the upsampled high-level features 'x'. The output then restores the tensor
        to the original height and width.

        :param x: high-level features, [batch, 2048, X, X], where X = input size / output stride
        :type x: torch.Tensor
        :param skip: low-level features, [batch, Y, 128, 128] where Y = 256 for ResNet, 128 for Xception
        :type skip: torch.Tensor
        :return: tensor with the final output, [batch, classes, input height, input width]
        :rtype: torch.Tensor
        """
        skip = self.low_level(skip)
        x = self.upsample(x)
        return self.output(torch.cat((skip, x), dim=1))


class DeepLabVariants(Enum):
    """Enum defining possible combinations of backbones and strides.
    Currently, only ResNet and Xception are supported as backbones.
    """
    RESNET50_16 = (rn.ResNetVariants.RN50, rn.OutputStrides.OS16, ASPPVariants.OS16)
    RESNET50_08 = (rn.ResNetVariants.RN50, rn.OutputStrides.OS08, ASPPVariants.OS08)
    RESNET101_16 = (rn.ResNetVariants.RN101, rn.OutputStrides.OS16, ASPPVariants.OS16)
    RESNET101_08 = (rn.ResNetVariants.RN101, rn.OutputStrides.OS08, ASPPVariants.OS08)
    XCEPTION08_16 = (xc.MiddleFlows.MF08, xc.OutputStrides.OS16, ASPPVariants.OS16)
    XCEPTION08_08 = (xc.MiddleFlows.MF08, xc.OutputStrides.OS08, ASPPVariants.OS08)
    XCEPTION16_16 = (xc.MiddleFlows.MF16, xc.OutputStrides.OS16, ASPPVariants.OS16)
    XCEPTION16_08 = (xc.MiddleFlows.MF16, xc.OutputStrides.OS08, ASPPVariants.OS08)


class DeepLabBase(nn.Module):
    """Generic DeepLab class that provides three inputs for the main block of the architecture,
    This provides a custom initialization when needed, otherwise it is advisable to use the specific
    V3 or V3Plus implementations."""

    def __init__(self,
                 in_channels: int,
                 in_dimension: int,
                 out_channels: int,
                 variant: DeepLabVariants = DeepLabVariants.RESNET101_16,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        super().__init__()
        assert out_channels > 0, "Please provide a valid number of classes!"
        backbone_variant, output_strides, aspp_variant = variant.value
        backbone_name = variant.name.lower()
        if backbone_name.startswith("resnet"):
            backbone = rn.ResNetBackbone(variant=backbone_variant,
                                         batch_norm=batch_norm,
                                         output_strides=output_strides,
                                         in_channels=in_channels)
        elif backbone_name.startswith("xception"):
            backbone = xc.XceptionBackbone(in_channels=in_channels,
                                           output_strides=output_strides,
                                           middle_flow=backbone_variant,
                                           batch_norm=batch_norm)
        output_dims = in_dimension // backbone.scaling_factor()
        features_high, _ = backbone.output_features()
        self.backbone = backbone
        self.aspp = ASPPModule(in_tensor=(features_high, output_dims, output_dims),
                               variant=aspp_variant,
                               batch_norm=batch_norm)


class DeepLabV3(DeepLabBase):
    """Deeplab V3 implementation: considering previous iterations V3 introduces a more modular
    concept of feature encoder, called 'backbone', and improves the ASPP module with more convolutions
    and global pooling. The CRF is also removed from the official implementation details.
    """

    def __init__(self,
                 in_channels: int = 3,
                 in_dimension: int = 512,
                 out_channels: int = 1,
                 variant: DeepLabVariants = DeepLabVariants.RESNET101_16,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        super().__init__(in_channels=in_channels,
                         in_dimension=in_dimension,
                         out_channels=out_channels,
                         variant=variant,
                         batch_norm=batch_norm)
        self.decoder = DecoderV3(output_stride=self.backbone.scaling_factor(),
                                 output_channels=out_channels,
                                 batch_norm=batch_norm)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Computes a forward pass on the whole DeepLab.
        In V3, the low-level features are not used by the other components.

        :param batch: input tensor, [batch, channels, height, width]
        :type batch: torch.Tensor
        :return: tensor with size [batch, classes, height, width]
        :rtype: torch.Tensor
        """
        x, _ = self.backbone(batch)
        x = self.aspp(x)
        return self.decoder(x)


class DeepLabV3Plus(DeepLabBase):
    """DeepLabV3Plus implementation, almost the same as V3, but with a much better decoding branch.
    """

    def __init__(self,
                 in_channels: int = 3,
                 in_dimension: int = 512,
                 out_channels: int = 1,
                 variant: DeepLabVariants = DeepLabVariants.XCEPTION16_16,
                 batch_norm: Type[nn.Module] = nn.BatchNorm2d):
        super().__init__(in_channels=in_channels,
                         in_dimension=in_dimension,
                         out_channels=out_channels,
                         variant=variant,
                         batch_norm=batch_norm)
        _, features_low = self.backbone.output_features()
        self.decoder = DecoderV3Plus(low_level_channels=features_low,
                                     output_stride=self.backbone.scaling_factor(),
                                     output_channels=out_channels,
                                     batch_norm=batch_norm)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Computes a forward pass on the whole network.
        In V3+, the low-level features are integrated for a high resolution mask.

        :param batch: input tensor, [batch, channels, height, width]
        :type batch: torch.Tensor
        :return: tensor with size [batch, classes, height, width]
        :rtype: torch.Tensor
        """
        x, skip = self.backbone(batch)
        x = self.aspp(x)
        return self.decoder(x, skip)


if __name__ == "__main__":
    x = torch.rand((2, 3, 480, 480))
    deeplab = DeepLabV3Plus(out_channels=10, in_dimension=480, variant=DeepLabVariants.XCEPTION08_16)
    print(deeplab(x).size())
