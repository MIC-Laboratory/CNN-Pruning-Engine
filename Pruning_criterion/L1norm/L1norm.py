import torch

def L1norm(layer):
    weight = layer.weight.data.clone()
    if len(weight.shape) == 4:
        importance = torch.sum(torch.abs(weight),dim=(1,2,3))
    else:
        
        importance = torch.sum(torch.abs(weight),dim=0)
    sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
    return sorted_idx