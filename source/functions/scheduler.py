import torch

def lr_scheduler(optimizer:torch.optim.Optimizer,
                 initial_lr:float,
                 steady_lr:float,
                 final_lr:float,
                 total_epochs:int,
                 start_epoch:int=0):
    
    increasing_epochs=int(total_epochs*0.1)
    decreasing_epochs=total_epochs-increasing_epochs

    increasing_scheduler=torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=initial_lr,
        end_factor=steady_lr,
        total_iters=increasing_epochs,
    )

    decreasing_scheduler=torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=steady_lr,
        end_factor=final_lr,
        total_iters=decreasing_epochs,
    )

    scheduler=torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[increasing_scheduler,decreasing_scheduler],
        milestones=[increasing_epochs],
        last_epoch=start_epoch-1
    )
    

    return optimizer,scheduler