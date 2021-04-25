import pytest
import torch


@pytest.fixture(scope="session")
def random_clf_batch():
    return torch.rand((2, 3, 224, 224))


@pytest.fixture(scope="session")
def random_seg_batch():
    return torch.rand((2, 3, 512, 512))
