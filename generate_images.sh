#!/bin/bash
#SBATCH -J generate_images
#SBATCH -o generate_images.out 
#SBATCH --no-requeue
#SBATCH --get-user-env
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1 # Number of GPUs per node
#SBATCH --mem=64G
#SBATCH --time=0:30:00   

source environment/bin/activate

echo "Start generating images"

srun python source/generate_images.py

echo "Done"



