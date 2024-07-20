import torch
import matplotlib.pyplot as plt
import time
import os
from torch.distributed import ReduceOp, reduce
from .losses import beta_gaussian_kldiv, mse_loss
from .schedulers import beta_scheduler

def train(rank, EPOCHS, train_loader,model,
           optimizer,scheduler,fraction=0,total_fractions=1):
    total_epochs=EPOCHS*total_fractions

    if rank==0:
        total_losses=[] if fraction==0 else torch.load('losses/total_losses.pt')
        reconstruction_losses=[] if fraction==0 else torch.load('losses/reconstruction_losses.pt')
        kl_divergences=[] if fraction==0 else torch.load('losses/kl_divergences.pt')
        start_time=time.time()

    for epoch in range(EPOCHS):
        actual_epoch=epoch+EPOCHS*fraction
        #beta=beta_scheduler(actual_epoch,total_epochs)
        beta=1
        train_epoch(model, optimizer,train_loader,rank,beta)

        model.eval()
        with torch.no_grad():
            evaluate(model,train_loader,rank, beta,
                     total_losses if rank==0 else None,reconstruction_losses if rank==0 else None,
                     kl_divergences if rank==0 else None)

            if rank==0:

                if epoch%10==0 or epoch==EPOCHS-1:
                    print(f"Epoch {actual_epoch}  Loss:{total_losses[-1]}",flush=True)
        
        scheduler.step()
        

    if rank==0:
        print(f"Completed training with loss: {total_losses[-1]}; Time: {(time.time()-start_time)/60}",flush=True)

        torch.save(model.module.state_dict(),'models/model.pt')

        if fraction!=total_fractions-1:
            torch.save(reconstruction_losses,'losses/reconstruction_losses.pt')
            torch.save(kl_divergences,'losses/kl_divergences.pt')
            torch.save(total_losses,'losses/total_losses.pt')
        else:
            plot_losses(reconstruction_losses,'reconstruction_losses')
            plot_losses(kl_divergences,'kl_divergences')
            plot_losses(total_losses,'total_losses')
            if total_fractions>1:
                os.remove('losses/reconstruction_losses.pt')
                os.remove('losses/kl_divergences.pt')
                os.remove('losses/total_losses.pt')

def train_epoch( model, optimizer,data_loader,device,beta=1):
    model.train()

    for _,data_point in enumerate(data_loader):
        image=data_point[0].to(device)
        label=data_point[1].to(device)

        generated_image,mu,log_var=model(image,label)
        loss=mse_loss(image,generated_image)+beta*beta_gaussian_kldiv(mu,log_var)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

def evaluate(model,data_loader,device, beta=1,reconstruction_losses=None,kl_divergences=None,total_losses=None):
    total_loss=0
    reconstruction_loss=0
    kl_divergence=0

    for _,data_point in enumerate(data_loader):
        image=data_point[0].to(device)
        label=data_point[1].to(device)

        generated_image,mu,log_var=model(image,label)
        single_reconstruction_loss=mse_loss(image,generated_image)
        single_kl_divergence=beta_gaussian_kldiv(mu,log_var)
        
        reconstruction_loss+=single_reconstruction_loss
        kl_divergence+=single_kl_divergence

        total_loss+=single_reconstruction_loss+beta*single_kl_divergence
    
    avg_reconstruction_loss=reconstruction_loss/len(data_loader)
    avg_kl_divergence=kl_divergence/len(data_loader)
    avg_total_loss=total_loss/len(data_loader)
    
    reduce(avg_reconstruction_loss,dst=0,op=ReduceOp.SUM)
    reduce(avg_kl_divergence,dst=0,op=ReduceOp.SUM)
    reduce(avg_total_loss,dst=0,op=ReduceOp.SUM)

    if total_losses is not None:
        reconstruction_losses.append(avg_reconstruction_loss.item())
        kl_divergences.append(avg_kl_divergence.item())
        total_losses.append(avg_total_loss.item())


def plot_losses(losses,name):
    plt.plot(losses,label='Losses')
    plt.xlabel("Epoch")
    plt.ylabel(str(name))
    plt.legend()
    plt.savefig('plots/'+str(name)+'_plot.png')
    plt.close()