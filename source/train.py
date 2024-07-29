import argparse
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group

from functions.vae import VariationalAutoEncoder
from functions.data_handling import prepare_data
from functions.ddp_setup import ddp_setup
from functions.data_handling import prepare_data
from functions.schedulers import lr_scheduler
from functions.trainer import Trainer

def main(device,world_size:int):

    IS_PARALLEL=torch.cuda.is_available() and world_size>1

    if IS_PARALLEL:
        ddp_setup(device,world_size,1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,default='/u/dssc/acesa000/fast/Street_View_Generator_data/hf_dataset_processed.pt')
    parser.add_argument('--fraction',type=int,default=0)
    parser.add_argument('--total_fractions',type=int,default=1)
    parser.add_argument('--checkpoint',type=str,default='None')
    parser.add_argument('--initial_learning_rate',type=float,default=1e-4)
    parser.add_argument('--steady_learning_rate',type=float,default=0.002)
    parser.add_argument('--final_learning_rate',type=float,default=1e-6)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch_size',type=int,default=512)
    parser.add_argument('--latent_dim',type=int,default=256)

    args = parser.parse_args()

    DATA_PATH=args.data_path
    FRACTION=args.fraction
    TOTAL_FRACTIONS=args.total_fractions
    CHECKPOINT_PATH=None if args.checkpoint=='None' else args.checkpoint
    INITIAL_LR = args.initial_learning_rate
    STEADY_LR = args.steady_learning_rate
    FINAL_LR = args.final_learning_rate
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LATENT_DIM = args.latent_dim

    data_loader= prepare_data(DATA_PATH,world_size, BATCH_SIZE,IS_PARALLEL)

    model=VariationalAutoEncoder(latent_dim=LATENT_DIM).to(device)

    if IS_PARALLEL:
        model=DDP(model,device_ids=[device])

    if FRACTION!=0:
        checkpoint=torch.load(CHECKPOINT_PATH)
        model_state_dict=checkpoint['model_state_dict']
        if IS_PARALLEL:
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)
    else:
        checkpoint=None

    model_parameters=model.module.parameters() if IS_PARALLEL else model.parameters()

    optimizer = torch.optim.Adam(model_parameters, lr=1, weight_decay=0, betas=(0.9, 0.95))

    optimizer,scheduler=lr_scheduler(optimizer=optimizer,initial_lr=INITIAL_LR,steady_lr=STEADY_LR,final_lr=FINAL_LR,total_epochs=EPOCHS*TOTAL_FRACTIONS)

    if FRACTION!=0:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    trainer=Trainer(data_loader,model,optimizer,scheduler,device,FRACTION,TOTAL_FRACTIONS,IS_PARALLEL,CHECKPOINT_PATH,checkpoint)

    trainer.train(EPOCHS)
    destroy_process_group()

if __name__ =="__main__":
    if torch.cuda.is_available():
        world_size=torch.cuda.device_count()
        mp.spawn(main,args=(world_size,),nprocs=world_size)
    else:
        main('cpu',1)

