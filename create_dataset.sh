#!/bin/bash
#SBATCH --no-requeue
#SBATCH -J create_dataset
#SBATCH --get-user-env
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o create_dataset.out
#SBATCH --mem=0
#SBATCH --time=01:00:00

source environment/bin/activate

python -u source/create_dataset.py --raw_data_directory='/u/dssc/acesa000/fast/Street_View_Generator_data/hf_dataset/images/test' --output_directory='/u/dssc/acesa000/fast/Street_View_Generator_data/hf_dataset_processed.pt' --metadata_directory='/u/dssc/acesa000/fast/Street_View_Generator_data/hf_dataset/test.csv'
echo "done"
