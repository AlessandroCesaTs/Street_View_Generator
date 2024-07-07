import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from functions.country_name_handling import CountryNameHandler

base_directory='/u/dssc/acesa000/fast/Street_View_Generator_data/hf_dataset/images/test'
output_dir = '/u/dssc/acesa000/fast/Street_View_Generator_data/hf_dataset_processed.pt'

encoder=CountryNameHandler()

metadata=pd.read_csv("/u/dssc/acesa000/fast/Street_View_Generator_data/hf_dataset/test.csv")
metadata=metadata[['id','country']]
metadata['country_encoding']=metadata.apply(lambda row: encoder.get_country_from_code(row['country']), axis=1)
metadata.drop(['country'],axis=1,inplace=True)
metadata.dropna(inplace=True)

dataset=[]

for folder in [base_directory+"/00",base_directory+"/01",base_directory+"/02",base_directory+"/03",base_directory+"/04"]:
    images = os.listdir(folder)
    for image in images:
        image_path = os.path.join(folder, image)
        image_id=int(image[:-4])
        if image_id in metadata.loc[:,'id'].values:
            country_encoding=metadata[metadata['id']==image_id]['country_encoding'].values[0]
                
            image = Image.open(image_path)
            image=image.resize((128,128), Image.LANCZOS)
            image=np.array(image)
            # Convert the numpy array to a torch tensor
            # The image is in HWC format, we need CHW format for PyTorch
            image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize to [0, 1]
            # Create the data point with the image tensor and the encoding
            
            data_point = (image_tensor, country_encoding)
            dataset.append(data_point)
torch.save(dataset,output_dir)


        
    
