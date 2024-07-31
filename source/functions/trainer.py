import os
import csv
import torch
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
from torch.distributed import ReduceOp, reduce
from .losses import gaussian_kldiv, mse_loss

class Trainer():
    def __init__(self,data_loader,model,optimizer,scheduler,beta_scheduler,device,is_parallel=False,
                 resume_from_checkpoint=False,checkpoint=None,output_path=None):
        self.data_loader=data_loader
        self.model=model
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.beta_scheduler=beta_scheduler
        self.device=device
        self.is_parallel=is_parallel
        self.checkpoint=checkpoint

        self.num_batches=len(self.data_loader)
        self.start_time=time.time()
        
        if self.is_parallel and self.device!=0:
            self.is_master_rank=False
        else:
            self.is_master_rank=True

        if self.is_master_rank:
            os.makedirs(os.path.join(output_path, 'checkpoints'), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'models'), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'plots'), exist_ok=True)
            os.makedirs(os.path.join(output_path, 'losses'), exist_ok=True)

        self.checkpoint_path=os.path.join(output_path, 'checkpoints','checkpoint.pt') if self.is_master_rank else None
        self.model_path=os.path.join(output_path, 'models','model.pt') if self.is_master_rank else None
        self.plots_path=os.path.join(output_path, 'plots') if self.is_master_rank else None
        self.losses_file_path=os.path.join(output_path,'losses','losses.csv') if self.is_master_rank else None

        if self.is_master_rank and not resume_from_checkpoint:
            with open(self.losses_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch','Reconstruction Loss','KL Divergence','Total Loss'])

        if resume_from_checkpoint:
            self.load_checkpoint()
        else:
            self.total_losses=[] if self.is_master_rank else None
            self.reconstruction_losses=[] if self.is_master_rank else None
            self.kl_divergences=[] if self.is_master_rank else None
            self.last_epoch=0

    def train(self,EPOCHS):
        for epoch in range(self.last_epoch,self.last_epoch+EPOCHS):
            beta=self.beta_scheduler.get_beta(epoch)
            self.model.train()
            self.train_epoch(beta)

            self.model.eval()
            with torch.no_grad():
                self.evaluate(beta)

                if self.is_master_rank and (epoch%10==0 or epoch==self.last_epoch or epoch==self.last_epoch+EPOCHS-1):
                    self.record_losses(epoch)
            
            self.scheduler.step()

        if self.is_master_rank:
            print(f"Completed training; Time: {(time.time()-self.start_time)/60}",flush=True)
            self.save_checkpoint(EPOCHS)
                

    def train_epoch(self,beta):
        for _,data_point in enumerate(self.data_loader):
            image=data_point[0].to(self.device)
            label=data_point[1].to(self.device)

            generated_image,mu,log_var=self.model(image,label)
            loss=mse_loss(image,generated_image)+beta*gaussian_kldiv(mu,log_var)
            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()
    
    def evaluate(self,beta):
        reconstruction_loss=0
        kl_divergence=0

        for _,data_point in enumerate(self.data_loader):
            image=data_point[0].to(self.device)
            label=data_point[1].to(self.device)

            generated_image,mu,log_var=self.model(image,label)

            reconstruction_loss+=mse_loss(image,generated_image)
            kl_divergence+=gaussian_kldiv(mu,log_var)

        avg_reconstruction_loss=reconstruction_loss/self.num_batches
        avg_kl_divergence=kl_divergence/self.num_batches
        
        if self.is_parallel:
            reduce(avg_reconstruction_loss,dst=0,op=ReduceOp.SUM)
            reduce(avg_kl_divergence,dst=0,op=ReduceOp.SUM)

        if self.is_master_rank :
            self.reconstruction_losses.append(avg_reconstruction_loss.item())
            self.kl_divergences.append(avg_kl_divergence.item())
            self.total_losses.append(self.reconstruction_losses[-1]+beta*self.kl_divergences[-1])

    def save_checkpoint(self,epoch):
        state = {
            'model_state_dict': self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'reconstruction_losses': self.reconstruction_losses,
            'kl_divergences': self.kl_divergences,
            'total_losses': self.total_losses,
            'epoch':epoch
        }
        torch.save(state,self.checkpoint_path)
        torch.save(state['model_state_dict'],self.model_path)
        self.plot_losses(self.reconstruction_losses,'reconstruction_losses')
        self.plot_losses(self.kl_divergences,'kl_divergences')
        self.plot_losses(self.total_losses,'total_losses')
    
    def load_checkpoint(self):
        self.reconstruction_losses=self.checkpoint['reconstruction_losses'] if self.is_master_rank else None
        self.kl_divergences=self.checkpoint['kl_divergences'] if self.is_master_rank else None
        self.total_losses=self.checkpoint['total_losses'] if self.is_master_rank else None
        self.last_epoch=self.checkpoint['epoch']

    def plot_losses(self,losses,name):
        path=os.path.join(self.plots_path,str(name)+'_plot.png')
        plt.plot(losses,label=str(name))
        plt.xlabel("Epoch")
        plt.ylabel(str(name))
        plt.legend()
        plt.savefig(path)
        plt.close()

    def record_losses(self,epoch):
        with open (self.losses_file_path,mode='a',newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch,self.reconstruction_losses[-1],self.kl_divergences[-1],self.total_losses[-1]])
