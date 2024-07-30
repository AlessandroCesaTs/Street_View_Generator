#!/bin/bash
#SBATCH -J generate_images
#SBATCH -o generate_images.out 
#SBATCH --no-requeue
#SBATCH --get-user-env
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=0:05:00   

data_path=$1
model_path=$2
images_path=$3

source environment/bin/activate

echo "Start generating images"

srun python source/generate_images.py --data_path=$data_path --model_path=$model_path --images_path=$images_path

echo "Done"



