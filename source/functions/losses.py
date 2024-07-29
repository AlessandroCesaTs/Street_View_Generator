import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


@torch.jit.script
def pixelwise_bce_sum(input: torch.Tensor,target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(input,target,reduction="sum")

def mse_loss(input: torch.Tensor,target: torch.Tensor)-> torch.Tensor:
    return F.mse_loss(input,target,reduction='sum')

@torch.jit.script
def beta_gaussian_kldiv(mu: torch.Tensor,sigma: torch.Tensor)-> torch.Tensor:
    return 0.5*(torch.pow(mu,2)+torch.exp(sigma)-sigma-1).sum()


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        resnet = resnet18(weights='DEFAULT',progress=False)
        self.layers = nn.Sequential(*list(resnet.children())[:6]).eval()  # Extracting up to layer 6
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_resnet, y_resnet = self.layers(x), self.layers(y)
        loss = F.mse_loss(x_resnet, y_resnet)
        return loss
