import os
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
from functions.country_name_handling import CountryNameHandler

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_directory',type=str)
parser.add_argument('--output_directory',type=str)
parser.add_argument('--metadata_directory',type=str)

args = parser.parse_args()

raw_data_directory=args.raw_data_directory
output_directory=args.output_directory
metadatadata_directory=args.metadata_directory

encoder=CountryNameHandler()

metadata=pd.read_csv(metadatadata_directory)
metadata=metadata[['id','country']]
metadata['country_encoding']=metadata.apply(lambda row: encoder.get_country_from_code(row['country']), axis=1)
metadata.drop(['country'],axis=1,inplace=True)
metadata.dropna(inplace=True)

dataset=[]

for folder in [raw_data_directory+"/00",raw_data_directory+"/01",raw_data_directory+"/02",raw_data_directory+"/03",raw_data_directory+"/04"]:
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
torch.save(dataset,output_directory)

print(f"Dataset preprocessed, it's in directory {output_directory}")


        
    
