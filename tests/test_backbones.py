import torch
from cvmodels.segmentation.backbones import resnet as rn, xception as xc


def test_resnet_out(random_seg_batch: torch.Tensor):
    batch, _, height, width = random_seg_batch.shape
    factors = [16, 8]
    high_features = [512, 512, 2048, 2048, 2048]
    low_features = [64, 64, 256, 256, 256]
    for i, stride in enumerate(rn.OutputStrides):
        for j, variant in enumerate(rn.ResNetVariants):
            model = rn.ResNetBackbone(variant=variant, output_strides=stride, pretrained=False)
            model.eval()
            with torch.no_grad():
                out = model(random_seg_batch)
                assert len(out) == 2
                high, low = out
                assert high.shape == (batch, high_features[j], height / factors[i], width / factors[i])
                assert low.shape == (batch, low_features[j], height / 4, width / 4)


def test_resnet_pretrain(random_seg_batch: torch.Tensor):
    batch, _, height, width = random_seg_batch.shape
    factors = [16, 8]
    high_features = [512, 512, 2048, 2048, 2048]
    low_features = [64, 64, 256, 256, 256]
    for i, stride in enumerate(rn.OutputStrides):
        for j, variant in enumerate(rn.ResNetVariants):
            model = rn.ResNetBackbone(variant=variant, output_strides=stride, pretrained=True)
            model.eval()
            with torch.no_grad():
                out = model(random_seg_batch)
                assert len(out) == 2
                high, low = out
                assert high.shape == (batch, high_features[j], height / factors[i], width / factors[i])
                assert low.shape == (batch, low_features[j], height / 4, width / 4)


def test_resnet_custom():
    random_batch = torch.rand((2, 4, 512, 512))
    batch, _, height, width = random_batch.shape
    factors = [16, 8]
    high_features = [512, 512, 2048, 2048, 2048]
    low_features = [64, 64, 256, 256, 256]
    for i, stride in enumerate(rn.OutputStrides):
        for j, variant in enumerate(rn.ResNetVariants):
            model = rn.ResNetBackbone(in_channels=4, variant=variant, output_strides=stride, pretrained=True)
            model.eval()
            with torch.no_grad():
                out = model(random_batch)
                assert len(out) == 2
                high, low = out
                assert high.shape == (batch, high_features[j], height / factors[i], width / factors[i])
                assert low.shape == (batch, low_features[j], height / 4, width / 4)


def test_xception_out(random_seg_batch: torch.Tensor):
    model = xc.XceptionBackbone(variant=xc.XceptionVariants.MF08, output_strides=xc.OutputStrides.OS16)
    model.eval()
    with torch.no_grad():
        out = model(random_seg_batch)
        assert len(out) == 2
        high, low = out
        assert high.shape == (2, 2048, 32, 32)
        assert low.shape == (2, 128, 128, 128)


def test_xception_pretrain(random_seg_batch: torch.Tensor):
    model = xc.XceptionBackbone(variant=xc.XceptionVariants.MF08,
                                output_strides=xc.OutputStrides.OS16,
                                pretrained=True)
    model.eval()
    with torch.no_grad():
        out = model(random_seg_batch)
        assert len(out) == 2
        high, low = out
        assert high.shape == (2, 2048, 32, 32)
        assert low.shape == (2, 128, 128, 128)


def test_xception_custom():
    random_batch = torch.rand((2, 4, 512, 512))
    model = xc.XceptionBackbone(in_channels=4,
                                variant=xc.XceptionVariants.MF08,
                                output_strides=xc.OutputStrides.OS16,
                                pretrained=True)
    model.eval()
    with torch.no_grad():
        out = model(random_batch)
        assert len(out) == 2
        high, low = out
        assert high.shape == (2, 2048, 32, 32)
        assert low.shape == (2, 128, 128, 128)
