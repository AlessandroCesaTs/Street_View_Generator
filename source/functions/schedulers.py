import torch

def lr_scheduler(optimizer:torch.optim.Optimizer,
                 initial_lr:float,
                 steady_lr:float,
                 final_lr:float,
                 total_epochs:int):
    
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
    )
    
    return optimizer,scheduler

class BetaScheduler():
    def __init__(self,initial_beta,final_beta,total_epochs):
        self.initial_beta=initial_beta
        self.factor=(final_beta-initial_beta)/total_epochs

    def get_beta(self,epoch):
        return self.initial_beta+self.factor*epoch