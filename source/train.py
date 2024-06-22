import argparse
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group

from functions.vae import *
from functions.data_handling import *
from functions.losses import *
from functions.ddp_setup import ddp_setup
from functions.data_handling import prepare_data
from functions.training_functions import train

def main(rank:int,world_size:int):

    ddp_setup(rank,world_size)

    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate',type=float,default=0.001)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--latent_dim',type=int,default=128)

    args = parser.parse_args()
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LATENT_DIM = args.latent_dim

    train_loader, validation_loader = prepare_data('/u/dssc/acesa000/fast/Street_View_Generator_data/dataset.pt',world_size, BATCH_SIZE)

    model=VariationalAutoEncoder(LATENT_DIM).to(rank)
    model=DDP(model,device_ids=[rank])
    optimizer=optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=0)

    train(rank, LEARNING_RATE, EPOCHS, LATENT_DIM, train_loader,
           validation_loader, model, optimizer)
    destroy_process_group()

if __name__ =="__main__":
    world_size=torch.cuda.device_count()
    mp.spawn(main,args=(world_size,),nprocs=world_size)

