import torch
import matplotlib.pyplot as plt
import time
from torch.distributed import ReduceOp, reduce
from .losses import beta_gaussian_kldiv, mse_loss

def train(rank, LEARNING_RATE, EPOCHS, LATENT_DIM, train_loader,
          validation_loader, model, optimizer):

    if rank==0:
        train_losses=[]
        validation_losses=[]
        lowest_validation_loss=float('inf')
        lowest_validation_loss_epoch=0
        best_weights=model.module.state_dict()
        start_time=time.time()

    for epoch in range(EPOCHS):
        train_epoch(model, optimizer,train_loader,rank)

        model.eval()
        with torch.no_grad():
            evaluate(model,train_loader,rank, 
                     train_losses if rank==0 else None)

            evaluate(model,validation_loader, rank,
                      validation_losses if rank==0 else None)

            if rank==0:
                if validation_losses[-1]<lowest_validation_loss:
                    lowest_validation_loss=validation_losses[-1]
                    lowest_validation_loss_epoch=epoch
                    best_weights=model.module.state_dict()

                if epoch%10==0 or epoch==EPOCHS-1:
                    print(f"Epoch {epoch}  Loss:{validation_losses[-1]}",flush=True)

    if rank==0:
        print(f"Completed training with lowest loss: {lowest_validation_loss} reached at EPOCH: {lowest_validation_loss_epoch}; Time: {(time.time()-start_time)/60}",flush=True)

        torch.save(best_weights,'models/model'+str(LEARNING_RATE)+'_'+str(LATENT_DIM)+'.pt')

        plot_losses(LEARNING_RATE, LATENT_DIM, train_losses, validation_losses)


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


def plot_losses(LEARNING_RATE, LATENT_DIM, train_losses, validation_losses):
    plt.plot(train_losses,label='Train')
    plt.plot(validation_losses,label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.legend()
    plt.savefig('plots/losses_plot_'+str(LEARNING_RATE)+'_'+str(LATENT_DIM)+'.png')