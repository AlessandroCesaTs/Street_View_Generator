from torch.utils.data import Dataset

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
    