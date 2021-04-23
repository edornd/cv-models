from typing import Tuple
from torch import Tensor

from cvmodels import ModuleBase


class Backbone(ModuleBase):
    """Definition for every backbone module
    """

    def scaling_factor(self) -> int:
        """Gets the output stride for the current model.

        :return: scaling factor of the backbone, from start to end
        :rtype: int
        """
        return 0

    def output_features(self) -> Tuple[int, int]:
        """Gets the final output sizes for the current backbone, both low and high dimensionality.

        :return: tuple representing the number of feature maps in output in the high and low level layers
        :rtype: Tuple[int, int]
        """
        return 0

    def forward(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass not implemented

        :param batch: input tensor, typically [batch, channels, height, width]
        :type batch: Tensor
        :return: usualy a backbone returns a set of tensors, in deeplab at most 2
        :rtype: Tuple[Tensor, Tensor]
        """
        return super().forward(batch)
