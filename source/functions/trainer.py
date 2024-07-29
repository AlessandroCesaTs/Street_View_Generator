import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
import time
from torch.distributed import ReduceOp, reduce
from .losses import beta_gaussian_kldiv, mse_loss,PerceptualLoss
from functions.schedulers import lr_scheduler


class Trainer():
    def __init__(self,data_loader,model,optimizer,scheduler,device,
                 fraction=0,total_fractions=1,is_parallel=False,checkpoint_path=None,checkpoint=None):
        self.data_loader=data_loader
        self.model=model
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.fraction=fraction
        self.total_fractions=total_fractions
        self.device=device
        self.is_parallel=is_parallel
        self.checkpoint_path=checkpoint_path
        self.checkpoint=checkpoint

        self.perceptual_loss_function=PerceptualLoss().to(self.device)
        self.beta=1
        self.num_batches=len(self.data_loader)
        self.start_time=time.time()

        model=DDP(model,device_ids=[device])

        if self.is_parallel and self.device!=0:
            self.is_master_rank=False
        else:
            self.is_master_rank=True

        if fraction==0:
            self.total_losses=[] if self.is_master_rank else None
            self.reconstruction_losses=[] if self.is_master_rank else None
            self.kl_divergences=[] if self.is_master_rank else None
            self.perceptual_losses=[] if self.is_master_rank else None
        else:
            self.load_checkpoint()

    def train(self,EPOCHS):
        for epoch in range(EPOCHS):
            actual_epoch=epoch+EPOCHS*self.fraction
            self.train_epoch()

            self.model.eval()
            with torch.no_grad():
                self.evaluate()

                if self.is_master_rank and (epoch%10==0 or epoch==EPOCHS-1):
                    print(f"Epoch {actual_epoch}  Loss:{self.total_losses[-1]}",flush=True)
            
            self.scheduler.step()

        if self.is_master_rank:
            print(f"Completed training with loss: {self.total_losses[-1]}; Time: {(time.time()-self.start_time)/60}",flush=True)

            if self.fraction!=self.total_fractions-1:
                self.save_checkpoint()
            else:
                self.plot_losses(self.reconstruction_losses,'reconstruction_losses')
                self.plot_losses(self.kl_divergences,'kl_divergences')
                self.plot_losses(self.perceptual_losses,'perceptual_losses')
                self.plot_losses(self.total_losses,'total_losses')

    def train_epoch(self):
        for _,data_point in enumerate(self.data_loader):
            image=data_point[0].to(self.device)
            label=data_point[1].to(self.device)

            generated_image,mu,log_var=self.model(image,label)
            loss=mse_loss(image,generated_image)+self.beta*beta_gaussian_kldiv(mu,log_var)+self.perceptual_loss_function(image,generated_image)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()
    
    def evaluate(self):
        reconstruction_loss=0
        kl_divergence=0
        perceptual_loss=0

        for _,data_point in enumerate(self.data_loader):
            image=data_point[0].to(self.device)
            label=data_point[1].to(self.device)

            generated_image,mu,log_var=self.model(image,label)

            reconstruction_loss+=mse_loss(image,generated_image)
            kl_divergence+=beta_gaussian_kldiv(mu,log_var)
            perceptual_loss+=self.perceptual_loss_function(image,generated_image)
        
        avg_reconstruction_loss=reconstruction_loss/self.num_batches
        avg_kl_divergence=kl_divergence/self.num_batches
        avg_perceptual_loss=perceptual_loss/self.num_batches
        
        if self.is_parallel:
            reduce(avg_reconstruction_loss,dst=0,op=ReduceOp.SUM)
            reduce(avg_kl_divergence,dst=0,op=ReduceOp.SUM)
            reduce(avg_perceptual_loss,dst=0,op=ReduceOp.SUM)

        if self.is_master_rank :
            print(f"reconstructuion {avg_reconstruction_loss}",flush=True)
            print(f"kl div {avg_kl_divergence}",flush=True)
            print(f"perceptual {avg_perceptual_loss}",flush=True)
            print(f"total {avg_reconstruction_loss+avg_kl_divergence+avg_perceptual_loss}",flush=True)

            self.reconstruction_losses.append(avg_reconstruction_loss.item())
            self.kl_divergences.append(avg_kl_divergence.item())
            self.perceptual_losses.append(avg_perceptual_loss.item())
            self.total_losses.append(self.reconstruction_losses[-1]+self.kl_divergences[-1]+self.perceptual_losses[-1])

    def plot_losses(self,losses,name):
        plt.plot(losses,label='Losses')
        plt.xlabel("Epoch")
        plt.ylabel(str(name))
        plt.legend()
        plt.savefig('plots/'+str(name)+'_plot.png')
        plt.close()

    def save_checkpoint(self):
        state = {
            'model_state_dict': self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'reconstruction_losses': self.reconstruction_losses,
            'kl_divergences': self.kl_divergences,
            'perceptual_losses': self.perceptual_losses,
            'total_losses': self.total_losses
        }
        torch.save(state, self.checkpoint_path)
    
    def load_checkpoint(self):
        self.reconstruction_losses=self.checkpoint['reconstruction_losses'] if self.is_master_rank else None
        self.kl_divergences=self.checkpoint['kl_divergences'] if self.is_master_rank else None
        self.perceptual_losses=self.checkpoint['perceptual_losses'] if self.is_master_rank else None
        self.total_losses=self.checkpoint['total_losses'] if self.is_master_rank else None