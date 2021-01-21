from typing import Tuple
from torch import Tensor
import torch.nn as nn


class Backbone(nn.Module):
    """Definition for every backbone module
    """

    def scaling_factor(self) -> int:
        """Gets the output stride for the current model.

        :return: scaling factor of the backbone, from start to end
        :rtype: int
        """
        return 0

    def output_features(self) -> int:
        """Gets the final output size for the current backbone.

        :return: int representing the number of feature maps in output
        :rtype: int
        """
        return 0

    def forward(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass not implemented

        :param batch: input tensor, typically [batch, channels, height, width]
        :type batch: Tensor
        :return: usualy a backbone returns a set of tensors, in deeplab at most 2
        :rtype: Tuple[Tensor]
        """
        return None
