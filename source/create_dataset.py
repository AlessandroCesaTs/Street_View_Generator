import os
import torch
import numpy as np
from functions.one_hot_encode import *
from PIL import Image


base_directory='/u/dssc/acesa000/fast/Street_View_Generator_data/selected_countries'
output_dir = '/u/dssc/acesa000/fast/Street_View_Generator_data/dataset.pt'

dataset=[]
directories= os.listdir(base_directory)
i=0
for country in directories:
    country_path = os.path.join(base_directory, country)
    encoding=one_hot_encode(country)
    images = os.listdir(country_path)
    for image in images:
        image_path = os.path.join(country_path, image)
        # Open the image file
        image = Image.open(image_path)
        image=image.resize((128,128), Image.LANCZOS)
        image=np.array(image)
        # Convert the numpy array to a torch tensor
        # The image is in HWC format, we need CHW format for PyTorch
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize to [0, 1]
        # Create the data point with the image tensor and the encoding
        data_point = (image_tensor, encoding)
        # Save the data point to the output directory
        dataset.append(data_point)
torch.save(dataset,output_dir)