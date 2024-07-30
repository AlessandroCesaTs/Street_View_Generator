import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

class CustomDataset(Dataset):
    def __init__(self,data):
        self.data=data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        data_point=self.data[idx]
        image=data_point[0]
        label=data_point[1]
        return image,label

def prepare_data(data_path,world_size, BATCH_SIZE,is_parallel):
    data=torch.load(data_path)[:100]
    dataset=CustomDataset(data)

    batch_size_per_process=int(BATCH_SIZE/world_size)
    if is_parallel:
        data_loader=DataLoader(dataset,batch_size=batch_size_per_process,shuffle=False,sampler=DistributedSampler(dataset))
    else:
        data_loader=DataLoader(dataset,batch_size=batch_size_per_process,shuffle=True)
    return data_loader
    
