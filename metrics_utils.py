from src.medifederate.xray import evaluate

def compute_metrics(model, dataloader, device='cpu'):
    return evaluate(model,dataloader,device)

def evaluate_model(model, dataloader, device='cpu'):
    return evaluate(model,dataloader,device)
