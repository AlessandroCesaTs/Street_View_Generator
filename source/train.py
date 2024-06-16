import argparse
import torch
import time
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from functions.vae import *
from functions.custom_dataset import *
from functions.losses import *

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate',type=float,default=0.001)
parser.add_argument('--epochs',type=int,default=100)
parser.add_argument('--batch_size',type=int,default=128)

device= torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


args = parser.parse_args()
learning_rate = args.learning_rate
EPOCHS = args.epochs
batch_size = args.batch_size

data=torch.load( '/u/dssc/acesa000/fast/Street_View_Generator_data/dataset.pt')
dataset=CustomDataset(data)

dataloader=DataLoader(dataset,batch_size=batch_size)

model=VariationalAutoEncoder().to(device)
optimizer=optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0)

losses=[]
lowest_loss=float('inf')
start_time=time.time()
print(device)

for epoch in range(EPOCHS):
    model.train()

    for _,data_point in enumerate(dataloader):
        image=data_point[0].to(device)
        label=data_point[1].to(device)

        generated_image,mu,log_var=model(image,label)
        loss=mse_loss(image,generated_image)+beta_gaussian_kldiv(mu,log_var)
        
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    model.eval()
    
    with torch.no_grad():
        validation_loss=0
        for _,data_point in enumerate(dataloader):
            image=data_point[0].to(device)
            label=data_point[1].to(device)

            generated_image,mu,log_var=model(image,label)
            validation_loss+=mse_loss(image,generated_image)+beta_gaussian_kldiv(mu,log_var)
        losses.append(validation_loss.item()/len(dataloader))
        if validation_loss<lowest_loss:
            lowest_loss=validation_loss
            lowest_loss_epoch=epoch
            best_weights=model.state_dict()
        if epoch%10==0 or epoch==EPOCHS-1:
            print(f"Epoch {epoch}  Loss:{loss}")

print(f"Completed training with lowest loss: {lowest_loss} reached at EPOCH: {lowest_loss_epoch}; Time: {time.time()-start_time}")

torch.save(best_weights,'models/model'+str(learning_rate)+'.pt')

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss")
plt.savefig('plots/losses_plot_'+str(learning_rate)+'.png')






