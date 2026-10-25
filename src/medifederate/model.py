from pathlib import Path
import torch
from torch import nn
from torchvision import models

def build_resnet18(num_classes=2, pretrained=True, dropout=0.2):
    weights=models.ResNet18_Weights.DEFAULT if pretrained else None
    m=models.resnet18(weights=weights)
    m.fc=nn.Sequential(nn.Dropout(dropout),nn.Linear(m.fc.in_features,num_classes))
    return m

def save_checkpoint(model,path,class_names):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state_dict":model.state_dict(),"class_names":class_names},p)

def load_checkpoint(path,device="cpu",dropout=0.2):
    c=torch.load(path,map_location=device)
    m=build_resnet18(len(c["class_names"]),False,dropout)
    m.load_state_dict(c["model_state_dict"])
    return m.to(device).eval(),c["class_names"]
