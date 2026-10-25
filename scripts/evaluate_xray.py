import json,torch
from src.medifederate.xray import loader,evaluate
from src.medifederate.model import load_checkpoint

def main():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,_=load_checkpoint('models/resnet18_xray.pt',device)
    dl,_=loader('data/processed/xray/test',batch_size=32)
    result=evaluate(model,dl,device)
    print(json.dumps(result,indent=2))
    with open('results/xray_metrics.json','w') as f: json.dump(result,f,indent=2)
if __name__=='__main__': main()
