#!/bin/bash
#SBATCH --no-requeue
#SBATCH -J download_data
#SBATCH --get-user-env
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o download_data.out
#SBATCH --mem=0
#SBATCH --time=01:00:00

source environment/bin/activate

kaggle datasets download -d ubitquitin/geolocation-geoguessr-images-50k -p /u/dssc/acesa000/fast/Street_View_Generator_data

echo "done"
