from enum import Enum
from typing import Type, Tuple
import torch
import torch.nn as nn

from cvmodels.segmentation.backbones import Backbone, resnet as rn


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
                                     nn.Conv2d(in_channels, 256, kernel_size=1, bias=False), batch_norm(256),
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


class DeepLab(nn.Module):
    """Generic DeepLab class that provides three inputs for the main block of the architecture,
    This provides a custom initialization when needed, otherwise it is advisable to use the specific
    V3 or V3Plus implementations.
    """

    def __init__(self, backbone: Backbone, aspp: nn.Module, decoder: nn.Module):
        """Creates a new deeplab network, where the three main components need to be provided as input
        The output stride of the submodules *must* match.

        :param backbone: standard headless CNN to be used as backbone (typically ResNet101)
        :type backbone: Type[Backbone]
        :param aspp: Atrous Spatial Pyramid Pooling, required to sample features at different scales
        :type aspp: Type[nn.Module]
        :param decoder: network head that transforms semantic information into a pixel-level mask
        :type decoder: nn.Module
        """
        super().__init__()
        self.backbone = backbone
        self.aspp = aspp
        self.decoder = decoder


class DeepLabVariants(Enum):
    RESNET50_16 = (rn.ResNetVariants.RN50, rn.OutputStrides.OS16, ASPPVariants.OS16)
    RESNET50_08 = (rn.ResNetVariants.RN50, rn.OutputStrides.OS08, ASPPVariants.OS08)
    RESNET101_16 = (rn.ResNetVariants.RN101, rn.OutputStrides.OS16, ASPPVariants.OS16)
    RESNET101_08 = (rn.ResNetVariants.RN101, rn.OutputStrides.OS08, ASPPVariants.OS08)


class DeepLabV3(DeepLab):
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
        backbone_variant, output_strides, aspp_variant = variant.value
        backbone = rn.ResNetBackbone(variant=backbone_variant,
                                     batch_norm=batch_norm,
                                     output_strides=output_strides,
                                     in_channels=in_channels)
        output_dims = in_dimension // backbone.scaling_factor()
        output_features = backbone.output_features()
        aspp = ASPPModule(in_tensor=(output_features, output_dims, output_dims),
                          variant=aspp_variant,
                          batch_norm=batch_norm)
        decoder = DecoderV3(output_stride=backbone.scaling_factor(),
                            output_channels=out_channels,
                            batch_norm=batch_norm)
        super().__init__(backbone, aspp, decoder)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        x, _ = self.backbone(batch)
        x = self.aspp(x)
        return self.decoder(x)


if __name__ == "__main__":
    x = torch.rand((2, 3, 512, 512))
    deeplab = DeepLabV3(out_channels=10)
    print(deeplab(x).size())
