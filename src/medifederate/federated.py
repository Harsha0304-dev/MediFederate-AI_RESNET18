import copy,io,time
import numpy as np
import torch
from torch import nn
from .model import build_resnet18
from .xray import loader,evaluate

def dirichlet_partition(labels,clients,alpha=0.30,seed=42):
    labels=np.asarray(labels); rng=np.random.default_rng(seed); result=[[] for _ in range(clients)]
    for cls in np.unique(labels):
        idx=np.where(labels==cls)[0]; rng.shuffle(idx)
        proportions=rng.dirichlet(np.full(clients,alpha))
        counts=np.floor(proportions*len(idx)).astype(int)
        counts[-1]=len(idx)-counts[:-1].sum()
        start=0
        for c,n in enumerate(counts): result[c].extend(idx[start:start+n].tolist()); start+=n
    for i in range(clients):
        if not result[i]:
            donor=max(range(clients),key=lambda k:len(result[k]))
            result[i].append(result[donor].pop())
    return result

def apply_dp(global_state,local_state,clip=1.0,noise=0.1,clients=3):
    keys=list(global_state.keys())
    delta=torch.cat([(local_state[k].float()-global_state[k].float()).flatten().cpu() for k in keys])
    norm=torch.linalg.vector_norm(delta).item(); scale=min(1.0,clip/(norm+1e-12)); out={}
    for k in keys:
        d=(local_state[k].float()-global_state[k].float())*scale
        std=noise*clip/max(clients,1)
        n=torch.normal(0.0,std,size=d.shape,device=d.device) if std>0 else torch.zeros_like(d)
        out[k]=(global_state[k].float()+d+n).to(local_state[k].dtype)
    return out

def pack(state,bits=8):
    if bits!=8: raise ValueError('Only 8-bit quantization is implemented.')
    out={}; total=0
    for k,v in state.items():
        x=v.float().cpu(); scale=max(float(x.abs().max())/127.0,1e-12)
        q=torch.clamp(torch.round(x/scale),-128,127).to(torch.int8)
        out[k]=(q,scale,v.dtype); total += q.numel()+8
    return out,total

def unpack(packed): return {k:(q.float()*scale).to(dtype) for k,(q,scale,dtype) in packed.items()}

def _serialized_size(state):
    b=io.BytesIO(); torch.save(state,b); return len(b.getvalue())

def fedavg(states,weights):
    total=float(sum(weights)); out=copy.deepcopy(states[0])
    for k in out:
        out[k]=sum(s[k].float()*float(w) for s,w in zip(states,weights))/total
    return out

def run(train_dir,test_dir,clients=3,rounds=5,non_iid=False,dp_noise=0,compression=False,device='cpu',batch_size=32,local_epochs=1,seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    _,ds=loader(train_dir,batch_size=batch_size); test,_=loader(test_dir,batch_size=batch_size)
    model=build_resnet18(2,True).to(device)
    if non_iid: parts=dirichlet_partition(ds.targets,clients,seed=seed)
    else:
        rng=np.random.default_rng(seed); idx=np.arange(len(ds)); rng.shuffle(idx); parts=[x.tolist() for x in np.array_split(idx,clients)]
    history=[]; communication=0; uncompressed_communication=0
    for r in range(rounds):
        states=[]; weights=[]; start=time.time()
        global_state=copy.deepcopy(model.state_dict())
        for ids in parts:
            dl=torch.utils.data.DataLoader(torch.utils.data.Subset(ds,ids),batch_size=batch_size,shuffle=True)
            local=copy.deepcopy(model); opt=torch.optim.AdamW(local.parameters(),lr=1e-4,weight_decay=1e-4); crit=nn.CrossEntropyLoss(); local.train()
            for _ in range(local_epochs):
                for x,y in dl:
                    opt.zero_grad(); loss=crit(local(x.to(device)),y.to(device)); loss.backward(); opt.step()
            state=local.state_dict()
            if dp_noise>0: state=apply_dp(global_state,state,1.0,dp_noise,clients)
            raw_bytes=_serialized_size(state)*2
            uncompressed_communication += raw_bytes
            if compression:
                packed,n=pack(state,8); state=unpack(packed); communication += n*2
            else: communication += raw_bytes
            states.append(state); weights.append(len(ids))
        model.load_state_dict(fedavg(states,weights)); met=evaluate(model,test,device)
        met.update(round=r+1,round_seconds=time.time()-start)
        history.append(met); print(f"Round {r+1}/{rounds}: accuracy={met['accuracy']:.4f} f1={met['f1']:.4f} roc_auc={met['roc_auc']}")
    reduction=(1-communication/uncompressed_communication) if uncompressed_communication else 0
    return history,{'communication_bytes':int(communication),'uncompressed_bytes':int(uncompressed_communication),'communication_reduction':float(reduction)}
