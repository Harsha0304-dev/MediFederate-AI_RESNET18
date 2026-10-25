import torch
from src.medifederate.xray import train_resnet

def main():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    train_resnet('data/processed/xray/train','data/processed/xray/validation','models/resnet18_xray.pt',device=device)
if __name__=='__main__': main()
