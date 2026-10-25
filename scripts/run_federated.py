import argparse,json,torch
from src.medifederate.federated import run

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rounds',type=int,default=5); p.add_argument('--clients',type=int,default=3)
    p.add_argument('--iid',action='store_true'); p.add_argument('--non-iid',action='store_true')
    p.add_argument('--dp-noise',type=float,default=0); p.add_argument('--compression',action='store_true')
    p.add_argument('--label',default='federated_results')
    a=p.parse_args()
    non_iid=a.non_iid or not a.iid
    device='cuda' if torch.cuda.is_available() else 'cpu'
    history,communication=run('data/processed/xray/train','data/processed/xray/test',a.clients,a.rounds,non_iid,a.dp_noise,a.compression,device)
    out={'history':history,**communication,'model':'ResNet18','non_iid':non_iid,'dp_noise':a.dp_noise,'compression':a.compression}
    print(json.dumps(out,indent=2))
    with open(f'results/{a.label}.json','w') as f: json.dump(out,f,indent=2)
if __name__=='__main__': main()
