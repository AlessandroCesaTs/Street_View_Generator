import argparse
import torch
import time
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import random_split,DataLoader
from functions.vae import *
from functions.custom_dataset import *
from functions.losses import *

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate',type=float,default=0.001)
parser.add_argument('--epochs',type=int,default=100)
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--latent_dim',type=int,default=128)


device= torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


args = parser.parse_args()
learning_rate = args.learning_rate
EPOCHS = args.epochs
batch_size = args.batch_size
latent_dim = args.latent_dim


max_beta=2
beta_growing_fraction=0.8

data=torch.load( '/u/dssc/acesa000/fast/Street_View_Generator_data/dataset.pt')
dataset=CustomDataset(data)
train_set,validation_set=random_split(dataset,[0.8,0.2])

train_loader=DataLoader(train_set,batch_size=batch_size)
validation_loader=DataLoader(validation_set,batch_size=batch_size)

model=VariationalAutoEncoder(latent_dim).to(device)
optimizer=optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0)

train_losses=[]
validation_losses=[]
lowest_validation_loss=float('inf')
lowest_validation_loss_epoch=0
best_weights=model.state_dict()
start_time=time.time()

for epoch in range(EPOCHS):

    #beta=compute_beta(EPOCHS,epoch,max_beta=max_beta,beta_growing_fraction=beta_growing_fraction)

    model.train()

    for _,data_point in enumerate(train_loader):
        image=data_point[0].to(device)
        label=data_point[1].to(device)

        generated_image,mu,log_var=model(image,label)
        loss=mse_loss(image,generated_image)+beta_gaussian_kldiv(mu,log_var)
        
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    model.eval()
    
    with torch.no_grad():
        train_loss=0

        for _,data_point in enumerate(train_loader):
            image=data_point[0].to(device)
            label=data_point[1].to(device)

            generated_image,mu,log_var=model(image,label)
            train_loss+=mse_loss(image,generated_image)+beta_gaussian_kldiv(mu,log_var)
        avg_train_loss=train_loss.item()/len(train_loader)
        train_losses.append(avg_train_loss)

        validation_loss=0
        for _,data_point in enumerate(validation_loader):
            image=data_point[0].to(device)
            label=data_point[1].to(device)

            generated_image,mu,log_var=model(image,label)
            validation_loss+=mse_loss(image,generated_image)+beta_gaussian_kldiv(mu,log_var)
        avg_validation_loss=validation_loss.item()/len(validation_loader)
        validation_losses.append(avg_validation_loss)
        
        if avg_validation_loss<lowest_validation_loss:
            lowest_validation_loss=avg_validation_loss
            lowest_validation_loss_epoch=epoch
            best_weights=model.state_dict()
        
        if epoch%10==0 or epoch==EPOCHS-1:
            print(f"Epoch {epoch}  Loss:{avg_validation_loss}")

print(f"Completed training with lowest loss: {lowest_validation_loss} reached at EPOCH: {lowest_validation_loss_epoch}; Time: {(time.time()-start_time)/60}")

torch.save(best_weights,'models/model'+str(learning_rate)+'_'+str(latent_dim)+'.pt')

plt.plot(train_losses,label='Train')
plt.plot(validation_losses,label='Validation')
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss")
plt.legend()
plt.savefig('plots/losses_plot_'+str(learning_rate)+'_'+str(latent_dim)+'.png')
