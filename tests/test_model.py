import torch
from src.medifederate.model import build_resnet18
def test_resnet():
    assert build_resnet18(2,False)(torch.randn(2,3,224,224)).shape==(2,2)
