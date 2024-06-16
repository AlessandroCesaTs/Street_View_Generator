#!/bin/bash
#SBATCH --no-requeue
#SBATCH -J train
#SBATCH --get-user-env
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=#1
#SBATCH --exclusive
#SBATCH -o train.out
#SBATCH --mem=0
#SBATCH --time=01:00:00

source environment/bin/activate

python -u source/train.py --epochs=3 --batch_size=1024

echo "done"
