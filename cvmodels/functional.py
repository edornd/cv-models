import torch
from torch.nn import functional as fn


def fixed_padding(inputs: torch.Tensor, kernel_size: int, dilation: int) -> torch.Tensor:
    """Computes the effective kernel size given the dilation (3x3 with dilation 2 is in fact a 5x5),
    then computes the start and end padding (e.g. for a 5x5: |xx|c|xx| => padding=2 left and right required),
    lastly it expands the input by the computed amount in all directions.

    :param inputs: input tensor not yet padded
    :type inputs: torch.Tensor
    :param kernel_size: nominal kernel size, not accounting for dilations
    :type kernel_size: int
    :param dilation: kernel dilation
    :type dilation: int
    :return: padded input, with the same shape, except height and width
    :rtype: torch.Tensor
    """
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = fn.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs
