#!/bin/bash
#SBATCH --no-requeue
#SBATCH --get-user-env
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2 # Number of GPUs per node
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00

fraction=${1:-0}
total_fractions=${2:-1}
weights=${3:-None}

source environment/bin/activate

echo "started fraction $fraction out of $(($total_fractions - 1))"

srun python -u source/train.py --fraction=$fraction --total_fractions=$total_fractions --weights=$weights --epochs=100

echo "done fraction $fraction out of $(($total_fractions - 1))"

