import os
import argparse
import torch
from torch.optim import Adam
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group

from functions.vae import VariationalAutoEncoder
from functions.data_handling import prepare_data
from functions.ddp_setup import ddp_setup
from functions.data_handling import prepare_data
from functions.schedulers import lr_scheduler,BetaScheduler
from functions.trainer import Trainer

def main(device,world_size:int):

    IS_PARALLEL=torch.cuda.is_available() and world_size>1
    if IS_PARALLEL:
        ddp_setup(device,world_size,1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str)
    parser.add_argument('--fraction',type=int,default=0)
    parser.add_argument('--total_fractions',type=int,default=1)
    parser.add_argument('--output_path',type=str)
    parser.add_argument('--initial_learning_rate',type=float,default=1e-4)
    parser.add_argument('--steady_learning_rate',type=float,default=0.002)
    parser.add_argument('--final_learning_rate',type=float,default=1e-6)
    parser.add_argument('--initial_beta',type=float,default=0)
    parser.add_argument('--final_beta',type=float,default=1)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch_size',type=int,default=1024)
    parser.add_argument('--latent_dim',type=int,default=256)

    args = parser.parse_args()

    DATA_PATH=args.data_path
    FRACTION=args.fraction
    TOTAL_FRACTIONS=args.total_fractions
    OUTPUT_PATH=args.output_path
    INITIAL_LR = args.initial_learning_rate
    STEADY_LR = args.steady_learning_rate
    FINAL_LR = args.final_learning_rate
    INITIAL_BETA = args.initial_beta
    FINAL_BETA = args.final_beta
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LATENT_DIM = args.latent_dim

    RESUME_FROM_CHECKPOINT = FRACTION!=1

    data_loader= prepare_data(DATA_PATH,world_size, BATCH_SIZE,IS_PARALLEL)

    model=VariationalAutoEncoder(latent_dim=LATENT_DIM).to(device)
    
    if IS_PARALLEL:
        model=DDP(model,device_ids=[device])

    model_parameters=model.module.parameters() if IS_PARALLEL else model.parameters()

    optimizer = Adam(model_parameters, lr=1, weight_decay=0, betas=(0.9, 0.95))

    if RESUME_FROM_CHECKPOINT:
        checkpoint_path=os.path.join(OUTPUT_PATH, 'checkpoints','checkpoint.pt')
        checkpoint=torch.load(checkpoint_path)
        model_state_dict=checkpoint['model_state_dict']
        if IS_PARALLEL:
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)
    else:
        checkpoint=None

    optimizer,scheduler=lr_scheduler(optimizer=optimizer,initial_lr=INITIAL_LR,steady_lr=STEADY_LR,final_lr=FINAL_LR,total_epochs=EPOCHS*TOTAL_FRACTIONS)
    beta_scheduler=BetaScheduler(INITIAL_BETA,FINAL_BETA,EPOCHS*TOTAL_FRACTIONS)
    if RESUME_FROM_CHECKPOINT:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    trainer=Trainer(data_loader,model,optimizer,scheduler,beta_scheduler,device,IS_PARALLEL,RESUME_FROM_CHECKPOINT,checkpoint,OUTPUT_PATH)

    trainer.train(EPOCHS)
    if IS_PARALLEL:
        destroy_process_group()

if __name__ =="__main__":
    if torch.cuda.is_available():
        world_size=torch.cuda.device_count()
        mp.spawn(main,args=(world_size,),nprocs=world_size)
    else:
        main('cpu',1)