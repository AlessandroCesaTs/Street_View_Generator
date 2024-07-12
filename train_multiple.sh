#!/bin/bash

job1_id=$(sbatch -J train_0 -o train_0.out train.sh 0 2 | awk '{print $4}')

job2_id=$(sbatch --dependency=afterok:$job1_id -J train_1 -o train_1.out train.sh 1 2 /u/dssc/acesa000/Street_View_Generator/models/model.pt| awk '{print $4}')

echo "First job ID: $job1_id"
echo "Second job ID: $job2_id (dependent on first job completion)"
