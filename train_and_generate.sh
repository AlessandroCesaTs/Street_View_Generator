#!/bin/bash

dataset_folder_dir=${1:-"/u/dssc/acesa000/fast/Street_View_Generator_data"}
dataset_dir="${dataset_folder_dir}/dataset_processed.pt"
project_dir=$(pwd)
images_dir="${project_dir}/images"
model_dir="${project_dir}/models/model.pt"

job1_id=$(sbatch -J train_0 -o train_0.out train.sh  "$dataset_dir" "$project_dir" 1 2| awk '{print $4}')

job2_id=$(sbatch --dependency=afterok:$job1_id -J train_1 -o train_1.out train.sh "$dataset_dir" "$project_dir" 2 2| awk '{print $4}')

job3_id=$(sbatch --dependency=afterok:$job2_id generate_images.sh "$dataset_dir" "$model_dir" "$images_dir"| awk '{print $4}')

echo "Submitted jobs $job1_id, $job2_id, $job3_id"