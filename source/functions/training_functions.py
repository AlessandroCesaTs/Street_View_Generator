import torch
import matplotlib.pyplot as plt
import time
import os
from torch.distributed import ReduceOp, reduce
from .losses import beta_gaussian_kldiv, mse_loss

def train(rank, EPOCHS, train_loader,model,
           optimizer,scheduler,fraction=0,total_fractions=1):

    if rank==0:
        train_losses=[] if fraction==0 else torch.load('losses/losses.pt') 
        start_time=time.time()

    for epoch in range(EPOCHS):
        train_epoch(model, optimizer,train_loader,rank)

        model.eval()
        with torch.no_grad():
            evaluate(model,train_loader,rank, 
                     train_losses if rank==0 else None)

            if rank==0:

                if epoch%10==0 or epoch==EPOCHS-1:
                    print(f"Epoch {epoch+EPOCHS*fraction}  Loss:{train_losses[-1]}",flush=True)
        
        scheduler.step()
        

    if rank==0:
        print(f"Completed training with loss: {train_losses[-1]}; Time: {(time.time()-start_time)/60}",flush=True)

        torch.save(model.module.state_dict(),'models/model.pt')

        if fraction!=total_fractions-1:
            torch.save(train_losses,'losses/losses.pt')
        else:
            plot_losses(train_losses)
            os.remove('losses/losses.pt')


def train_epoch( model, optimizer,data_loader,device):
    model.train()

    for _,data_point in enumerate(data_loader):
        image=data_point[0].to(device)
        label=data_point[1].to(device)

        generated_image,mu,log_var=model(image,label)
        loss=mse_loss(image,generated_image)+beta_gaussian_kldiv(mu,log_var)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


def evaluate(model,data_loader,device, losses=None):
    train_loss=0

    for _,data_point in enumerate(data_loader):
        image=data_point[0].to(device)
        label=data_point[1].to(device)

        generated_image,mu,log_var=model(image,label)
        train_loss+=mse_loss(image,generated_image)+beta_gaussian_kldiv(mu,log_var)
    avg_loss=train_loss/len(data_loader)
    reduce(avg_loss,dst=0,op=ReduceOp.SUM)

    if losses is not None:
        losses.append(avg_loss.item())


def plot_losses(train_losses):
    plt.plot(train_losses,label='Train')
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.legend()
    plt.savefig('plots/losses_plot.png')