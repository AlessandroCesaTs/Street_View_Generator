import torch
from typing import List

def lr_scheduler(optimizer:torch.optim.Optimizer,
                 initial_lr:float,
                 steady_lr:float,
                 final_lr:float,
                 total_epochs:int,
                 start_epoch:int=0):
    
    warmup_epochs=int(total_epochs*0.1)
    steady_epochs=0
    anneal_epochs=total_epochs-warmup_epochs

    for grp in optimizer.param_groups:
        grp["lr"]=steady_lr

    warmup_scheduler=torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=initial_lr/steady_lr,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    steady_scheduler=torch.optim.lr_scheduler.ConstantLR(
        optimizer=optimizer,
        factor=1.0,
        total_iters=steady_epochs,
    )

    anneal_scheduler=torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=1.0,
        end_factor=final_lr/steady_lr,
        total_iters=anneal_epochs,
    )

    scheduler=torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler,steady_scheduler,anneal_scheduler],
        milestones=[warmup_epochs,warmup_epochs+steady_epochs],
    )
    
    scheduler.last_epoch=start_epoch-1

    return optimizer,scheduler