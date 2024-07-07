import torch
import torch.nn as nn
import torch.nn.functional as F
from .country_name_handling import CountryNameHandler

def make_conv_block(in_channels:int,
    out_channels:int,
    kernel_size:int,
    stride:int=1,
    padding:int=0,
    leaky_slope:float=0.2)-> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
        nn.BatchNorm2d(out_channels,affine=True),
        nn.LeakyReLU(leaky_slope )
    )

def make_deconv_block(in_channels:int,
    out_channels:int,
    kernel_size:int,
    stride:int=1,
    padding:int=0,
    output_padding:int=0,
    leaky_slope:float=0.2)-> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,output_padding,bias=False),
        nn.BatchNorm2d(out_channels,affine=True),
        nn.LeakyReLU(leaky_slope)
    )

def image_encoder():
    return nn.Sequential(
        make_conv_block(3,32,3,2,1), #128->64
        make_conv_block(32,64,3,2,1), #64->32
        make_conv_block(64,128,3,2,1), #32->16
        make_conv_block(128,256,3,2,1), #16->8
        nn.Flatten()
    )

def label_embedder(label_embedding_dim):
    return nn.Sequential(
        nn.Linear(154,label_embedding_dim),
        nn.LeakyReLU(negative_slope=0.2)
    )

def linear_with_label(latent_dim,label_embedding_dim):
    return nn.Sequential(
        nn.Linear(latent_dim+label_embedding_dim,256*8*8),
        nn.LeakyReLU(negative_slope=0.2)
    )

class LinearNeck(nn.Module):
    def __init__(self,latent_dim,label_embedding_dim):
        super().__init__()
        self.fc=nn.Linear(256*8*8+label_embedding_dim,latent_dim)
        self.to_mu=nn.Linear(latent_dim,latent_dim)
        self.to_log_var=nn.Linear(latent_dim,latent_dim)
    def forward(self,encoded_image,encoded_label):
        x=torch.cat((encoded_image,encoded_label),dim=1)
        x=F.relu(self.fc(x))
        mu=self.to_mu(x)
        log_var=self.to_log_var(x)
        return mu,log_var

class GaussianReparametrizerSampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,z_mu:torch.Tensor,z_log_var:torch.Tensor)->torch.Tensor:
        return z_mu+torch.randn_like(z_mu,device=z_mu.device)*torch.exp(z_log_var * 0.5)

def decoder():
    return nn.Sequential(
        make_deconv_block(256,256,3,1,1), #8->8
        make_deconv_block(256,128,3,2,1,1), #8->16
        make_deconv_block(128,64,3,2,1,1),  #16->32
        make_deconv_block(64,32,3,2,1,1),   #32->64
        nn.ConvTranspose2d(32,3,3,2,1,1,bias=True), #64->128
        nn.Sigmoid()
    )

class VariationalAutoEncoder(nn.Module):
    def __init__(self,latent_dim=256,label_embedding_dim=512):
        super().__init__()
        self.latent_dim=latent_dim
        self.label_embedding_dim=label_embedding_dim
        self.name_handler=CountryNameHandler()

        self.image_encoder=image_encoder()
        self.label_embedder=label_embedder(self.label_embedding_dim)
        self.linear_neck=LinearNeck(self.latent_dim,self.label_embedding_dim)
        self.sampler=GaussianReparametrizerSampler()
        self.linear_with_label=linear_with_label(self.latent_dim,self.label_embedding_dim)
        self.decoder=decoder()
    
    def forward(self,input_image:torch.Tensor,input_label:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        mu,log_var=self.linear_neck(self.image_encoder(input_image),self.label_embedder(input_label))
        latent=self.sampler(mu,log_var)
        x=torch.cat((latent,self.label_embedder(input_label)),dim=1)
        x=self.linear_with_label(x)
        x=x.view((-1,256,8,8))
        return self.decoder(x),mu,log_var
    
    def generate(self,input_label):
        encoded_label=self.name_handler.one_hot_encode(input_label)
        latent=torch.randn(self.latent_dim).to(encoded_label.device)
        x=torch.cat((latent,self.label_embedder(encoded_label)),dim=0)
        x=self.linear_with_label(x)
        x=x.view((-1,256,8,8))
        return self.decoder(x)
        

