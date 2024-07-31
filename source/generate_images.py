import os
import argparse
import torch
import matplotlib.pyplot as plt
from functions.vae import *

parser = argparse.ArgumentParser()

parser.add_argument('--data_path',type=str)
parser.add_argument('--model_path',type=str)
parser.add_argument('--images_path',type=str)

args = parser.parse_args()

data_path=args.data_path
images_path=args.images_path
model_path=args.model_path

os.makedirs(images_path, exist_ok=True)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

data=torch.load(data_path)

model=VariationalAutoEncoder(256).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

plt.imshow(data[0][0].permute(1,2,0))
plt.savefig(os.path.join(images_path,'input_image.png'))
plt.close()

output_image=model(data[0][0].unsqueeze(0).to(device),data[1][1].unsqueeze(0).to(device))[0]
plt.imshow(output_image[0].permute(1,2,0).detach().cpu().numpy() )
plt.savefig(os.path.join(images_path,'reconstructed_image.png'))
plt.close()

generated_usa_1=model.generate('United States')
plt.imshow(generated_usa_1[0].permute(1,2,0).detach().cpu().numpy() )
plt.savefig(os.path.join(images_path,'usa_1.png'))
plt.close()

generated_usa_2=model.generate('United States')
plt.imshow(generated_usa_2[0].permute(1,2,0).detach().cpu().numpy() )
plt.savefig(os.path.join(images_path,'usa_2.png'))
plt.close()

generated_italy=model.generate('Italy')
plt.imshow(generated_italy[0].permute(1,2,0).detach().cpu().numpy() )
plt.savefig(os.path.join(images_path,'italy.png'))
plt.close()

generated_morocco=model.generate('Morocco')
plt.imshow(generated_morocco[0].permute(1,2,0).detach().cpu().numpy() )
plt.savefig(os.path.join(images_path,'morocco.png'))
plt.close()