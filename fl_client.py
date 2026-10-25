"""Flower-compatible client placeholder/adapter.
The recorded experiments use the deterministic manual engine in src.medifederate.federated.
"""
from src.medifederate.model import build_resnet18

def create_model(num_classes=2):
    return build_resnet18(num_classes=num_classes, pretrained=True)
