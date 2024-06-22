import torch
from torch.utils.data import DataLoader, Dataset, random_split
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


def prepare_data(data_path,world_size, BATCH_SIZE):
    data=torch.load(data_path)
    dataset=CustomDataset(data)
    train_set,validation_set=random_split(dataset,[0.8,0.2])

    batch_size_per_process=int(BATCH_SIZE/world_size)

    train_loader=DataLoader(train_set,batch_size=batch_size_per_process,shuffle=False,sampler=DistributedSampler(train_set))
    validation_loader=DataLoader(validation_set,batch_size=batch_size_per_process,shuffle=False,sampler=DistributedSampler(validation_set))
    return train_loader,validation_loader
    