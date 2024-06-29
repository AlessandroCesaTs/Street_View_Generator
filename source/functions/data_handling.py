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

    batch_size_per_process=int(BATCH_SIZE/world_size)

    data_loader=DataLoader(dataset,batch_size=batch_size_per_process,shuffle=False,sampler=DistributedSampler(dataset))
    return data_loader
    