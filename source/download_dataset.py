import os
import argparse
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str)

args = parser.parse_args()

data_path=args.data_path

os.makedirs(data_path,exist_ok=True)

for i in range(5):
    hf_hub_download(repo_id="osv5m/osv5m", filename=str(i).zfill(2)+'.zip', subfolder="images/test", repo_type='dataset', local_dir=data_path)

hf_hub_download(repo_id="osv5m/osv5m", filename='test.csv',  repo_type='dataset', local_dir=data_path)

print(f"Dataset correctly downloaded in directory {data_path}")
