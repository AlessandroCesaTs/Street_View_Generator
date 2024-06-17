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

for learning_rate in 0.00005
do
    echo "started learning rate $learning_rate"
    python -u source/train.py --epochs=230 --learning_rate=$learning_rate --batch_size=1024
    echo "done learning rate $learning_rate"
done

echo "done"
