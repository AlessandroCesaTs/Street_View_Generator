import torch
import torch.nn as nn
import torch.nn.functional as F

def mse_loss(input: torch.Tensor,target: torch.Tensor)-> torch.Tensor:
    return F.mse_loss(input,target,reduction='sum')

@torch.jit.script
def gaussian_kldiv(mu: torch.Tensor,sigma: torch.Tensor)-> torch.Tensor:
    return 0.5*(torch.pow(mu,2)+torch.exp(sigma)-sigma-1).sum()
