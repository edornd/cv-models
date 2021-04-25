from typing import Any
import torch
import torch.nn as nn

from cvmodels.segmentation import unet, deeplab as dl


def output(model: nn.Module, input_batch: torch.Tensor) -> Any:
    model.eval()
    with torch.no_grad():
        return model(input_batch)


def numel(m: torch.nn.Module, only_trainable: bool = True) -> int:
    parameters = m.parameters()
    if only_trainable:
        parameters = list(p for p in parameters if p.requires_grad)
    unique = dict((p.data_ptr(), p) for p in parameters).values()
    return sum(p.numel() for p in unique)


def test_unet_out_transpose(random_seg_batch: torch.Tensor):
    batches, _, height, width = random_seg_batch.shape
    model = unet.UNet(bilinear=False, outputs=1)
    assert numel(model) > 31_000_000
    out = output(model, random_seg_batch)
    assert out.shape == (batches, 1, height, width)


def test_unet_out_bilinear(random_seg_batch: torch.Tensor):
    batches, _, height, width = random_seg_batch.shape
    model = unet.UNet(bilinear=True, outputs=1)
    assert numel(model) < 30_000_000
    out = output(model, random_seg_batch)
    assert out.shape == (batches, 1, height, width)


def test_deeplabv3_out(random_seg_batch: torch.Tensor):
    batches, _, height, width = random_seg_batch.shape
    for variant in dl.DeepLabVariants:
        model = dl.DeepLabV3(variant=variant)
        out = output(model, random_seg_batch)
        assert out.shape == (batches, 1, height, width)


def test_deeplabv3_pretrain_backbone(random_seg_batch: torch.Tensor):
    batches, _, height, width = random_seg_batch.shape
    for variant in dl.DeepLabVariants:
        model = dl.DeepLabV3(variant=variant, pretrained=True)
        out = output(model, random_seg_batch)
        assert out.shape == (batches, 1, height, width)


def test_deeplabv3_custom():
    batch = torch.rand((2, 4, 480, 480))
    batches, _, height, width = batch.shape
    for variant in dl.DeepLabVariants:
        model = dl.DeepLabV3(in_channels=4, out_channels=2, in_dimension=480, variant=variant, pretrained=True)
        out = output(model, batch)
        assert out.shape == (batches, 2, height, width)


def test_deeplabv3plus_out(random_seg_batch: torch.Tensor):
    batches, _, height, width = random_seg_batch.shape
    for variant in dl.DeepLabVariants:
        model = dl.DeepLabV3Plus(variant=variant)
        out = output(model, random_seg_batch)
        assert out.shape == (batches, 1, height, width)


def test_deeplabv3plus_pretrain_backbone(random_seg_batch: torch.Tensor):
    batches, _, height, width = random_seg_batch.shape
    for variant in dl.DeepLabVariants:
        model = dl.DeepLabV3Plus(variant=variant, pretrained=True)
        out = output(model, random_seg_batch)
        assert out.shape == (batches, 1, height, width)


def test_deeplabv3plus_custom():
    batch = torch.rand((2, 4, 480, 480))
    batches, _, height, width = batch.shape
    for variant in dl.DeepLabVariants:
        model = dl.DeepLabV3Plus(in_channels=4, out_channels=2, in_dimension=480, variant=variant, pretrained=True)
        out = output(model, batch)
        assert out.shape == (batches, 2, height, width)
