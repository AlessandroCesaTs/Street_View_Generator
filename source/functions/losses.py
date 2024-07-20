import torch
import torch.nn.functional as F

@torch.jit.script
def pixelwise_bce_sum(input: torch.Tensor,target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(input,target,reduction="sum")

def mse_loss(input: torch.Tensor,target: torch.Tensor)-> torch.Tensor:
    return F.mse_loss(input,target,reduction='sum')

@torch.jit.script
def beta_gaussian_kldiv(mu: torch.Tensor,sigma: torch.Tensor)-> torch.Tensor:
    return 0.5*(torch.pow(mu,2)+torch.exp(sigma)-sigma-1).sum()
