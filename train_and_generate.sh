#!/bin/bash

job1_id=$(sbatch -J train_0 -o train_0.out train.sh 0 2 '/u/dssc/acesa000/Street_View_Generator/checkpoints/checkpoint.pt'| awk '{print $4}')

job2_id=$(sbatch --dependency=afterok:$job1_id -J train_1 -o train_1.out train.sh 1 2 '/u/dssc/acesa000/Street_View_Generator/checkpoints/checkpoint.pt'| awk '{print $4}')

job3_id=$(sbatch --dependency=afterok:$job2_id generate_images.sh | awk '{print $4}')

echo "Submitted jobs $job1_id, $job2_id, $job3_id"

