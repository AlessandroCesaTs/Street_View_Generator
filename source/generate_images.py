import torch
import matplotlib.pyplot as plt
from functions.vae import *
import time

start_time=time.time()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

data=torch.load('/u/dssc/acesa000/fast/Street_View_Generator_data/hf_dataset_processed.pt')

model=VariationalAutoEncoder(256)
model.load_state_dict(torch.load('models/model.pt',map_location=device))
model.eval()

plt.imshow(data[0][0].permute(1,2,0))
plt.savefig('/u/dssc/acesa000/Street_View_Generator/generated_images/input_image.png')
plt.close()


output_image=model(data[0][0].unsqueeze(0),data[1][1].unsqueeze(0))[0]
plt.imshow(output_image[0].permute(1,2,0).detach().numpy() )
plt.savefig('/u/dssc/acesa000/Street_View_Generator/generated_images/output_image.png')
plt.close()

generated_usa_1=model.generate('United States')
plt.imshow(generated_usa_1[0].permute(1,2,0).detach().numpy() )
plt.savefig('/u/dssc/acesa000/Street_View_Generator/generated_images/generated_usa_1.png')
plt.close()

generated_usa_2=model.generate('United States')
plt.imshow(generated_usa_2[0].permute(1,2,0).detach().numpy() )
plt.savefig('/u/dssc/acesa000/Street_View_Generator/generated_images/generated_usa_2.png')
plt.close()

generated_italy=model.generate('Italy')
plt.imshow(generated_italy[0].permute(1,2,0).detach().numpy() )
plt.savefig('/u/dssc/acesa000/Street_View_Generator/generated_images/generated_italy.png')
plt.close()

generated_morocco=model.generate('Morocco')
plt.imshow(generated_morocco[0].permute(1,2,0).detach().numpy() )
plt.savefig('/u/dssc/acesa000/Street_View_Generator/generated_images/generated_morocco.png')
plt.close()

print(f"Total time: {(time.time()-start_time)/60}")







