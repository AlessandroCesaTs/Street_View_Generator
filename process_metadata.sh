#!/bin/bash
#SBATCH --no-requeue
#SBATCH -J process_metadata
#SBATCH --get-user-env
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o process_metadata.out
#SBATCH --time=01:00:00

source environment/bin/activate

python -u source/process_metadata.py
echo "done"
