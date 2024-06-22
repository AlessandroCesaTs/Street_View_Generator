#!/bin/bash
#SBATCH --no-requeue
#SBATCH -J train
#SBATCH --get-user-env
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2 # Number of GPUs per node
#SBATCH --exclusive
#SBATCH -o train.out
#SBATCH --mem=0
#SBATCH --time=01:00:00

source environment/bin/activate

for latent_dim in 256
do
    echo "started latent_dim $latent_dim"
    srun python -u source/train.py --epochs=225 --learning_rate=0.00005 --batch_size=1024 --latent_dim=$latent_dim
    echo "done latent_dim $latent_dim"
done

echo "done"

