#!/bin/bash
#SBATCH --no-requeue
#SBATCH --get-user-env
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=1:00:00

data_path=$1
output_path=$2
fraction=${3:-0}
total_fractions=${4:-1}

source environment/bin/activate

echo "started fraction $fraction out of $(($total_fractions))"

srun python -u source/train.py --fraction=$fraction --total_fractions=$total_fractions --data_path=$data_path --output_path=$output_path --epochs=100

echo "done fraction $fraction out of $(($total_fractions))"

