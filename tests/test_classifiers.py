import logging
import torch
import torch.nn as nn

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from cvmodels.classification import resnet as rn
from cvmodels.classification import xception as xc


log = logging.getLogger(__name__)


def compare(model_a: nn.Module, model_b: nn.Module, data: torch.Tensor, check_results: bool = True) -> None:
    sda = model_a.state_dict()
    sdb = model_b.state_dict()
    for k in sda:
        assert k in sdb
        log.debug(f"{k:<40s}", sda[k].shape, sdb[k].shape)
        assert sda[k].shape == sdb[k].shape
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        xa = model_a(data)
        xb = model_b(data)
        assert xa.shape == xb.shape
        if check_results:
            assert torch.all(xa == xb)
        return xa, xb


def test_resnet18_out(random_clf_batch: torch.Tensor):
    original = resnet18(pretrained=False)
    model = rn.ResNet18(pretrained=False)
    _, pred = compare(original, model, random_clf_batch, check_results=False)
    assert pred.shape == (2, 1000)
    original = resnet18(pretrained=True)
    model = rn.ResNet18(pretrained=True)
    _, pred = compare(original, model, random_clf_batch)
    assert pred.shape == (2, 1000)


def test_resnet18_custom():
    model = rn.ResNet18(in_channels=4, num_classes=10)
    x = torch.rand((2, 4, 224, 224))
    out = model(x)
    assert out.shape == (2, 10)


def test_resnet34_out(random_clf_batch: torch.Tensor):
    original = resnet34(pretrained=False)
    model = rn.ResNet34(pretrained=False)
    _, pred = compare(original, model, random_clf_batch, check_results=False)
    assert pred.shape == (2, 1000)
    original = resnet34(pretrained=True)
    model = rn.ResNet34(pretrained=True)
    _, pred = compare(original, model, random_clf_batch)
    assert pred.shape == (2, 1000)


def test_resnet34_custom():
    model = rn.ResNet34(in_channels=4, num_classes=10)
    x = torch.rand((2, 4, 224, 224))
    out = model(x)
    assert out.shape == (2, 10)


def test_resnet50_out(random_clf_batch: torch.Tensor):
    original = resnet50(pretrained=False)
    model = rn.ResNet50(pretrained=False)
    _, pred = compare(original, model, random_clf_batch, check_results=False)
    assert pred.shape == (2, 1000)
    original = resnet50(pretrained=True)
    model = rn.ResNet50(pretrained=True)
    _, pred = compare(original, model, random_clf_batch)
    assert pred.shape == (2, 1000)


def test_resnet50_custom():
    model = rn.ResNet50(in_channels=4, num_classes=10)
    x = torch.rand((2, 4, 224, 224))
    out = model(x)
    assert out.shape == (2, 10)


def test_resnet101_out(random_clf_batch: torch.Tensor):
    original = resnet101(pretrained=False)
    model = rn.ResNet101(pretrained=False)
    _, pred = compare(original, model, random_clf_batch, check_results=False)
    assert pred.shape == (2, 1000)
    original = resnet101(pretrained=True)
    model = rn.ResNet101(pretrained=True)
    _, pred = compare(original, model, random_clf_batch)
    assert pred.shape == (2, 1000)


def test_resnet101_custom():
    model = rn.ResNet101(in_channels=4, num_classes=10)
    x = torch.rand((2, 4, 224, 224))
    out = model(x)
    assert out.shape == (2, 10)


def test_resnet152_out(random_clf_batch: torch.Tensor):
    original = resnet152(pretrained=False)
    model = rn.ResNet152(pretrained=False)
    _, pred = compare(original, model, random_clf_batch, check_results=False)
    assert pred.shape == (2, 1000)
    original = resnet152(pretrained=True)
    model = rn.ResNet152(pretrained=True)
    _, pred = compare(original, model, random_clf_batch)
    assert pred.shape == (2, 1000)


def test_resnet152_custom():
    model = rn.ResNet152(in_channels=4, num_classes=10)
    x = torch.rand((2, 4, 224, 224))
    out = model(x)
    assert out.shape == (2, 10)


def test_xception_out(random_clf_batch: torch.Tensor):
    model = xc.Xception(pretrained=False)
    out1 = model(random_clf_batch)
    assert out1.shape == (2, 1000)
    model = xc.Xception(pretrained=True)
    out2 = model(random_clf_batch)
    assert out2.shape == (2, 1000)
    assert torch.any(out1 != out2)


def test_xception_custom():
    model = xc.Xception(in_channels=4, num_classes=10)
    x = torch.rand((2, 4, 299, 299))
    out = model(x)
    assert out.shape == (2, 10)
