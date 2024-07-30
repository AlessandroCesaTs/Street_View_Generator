#!/bin/bash
#SBATCH --no-requeue
#SBATCH -J create_dataset
#SBATCH --get-user-env
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o create_dataset.out
#SBATCH --mem=64G
#SBATCH --time=0:45:00

echo "Start creating the dataset"
project_dir=$(pwd)

data_path=$1
raw_data_directory="${data_path}/images/test"
output_directory="${data_path}/dataset_processed.pt"
metadata_directory="${data_path}/test.csv"

source environment/bin/activate

srun python source/download_dataset.py --data_path=$data_path

cd "$raw_data_directory"

echo "Extracting files"

for i in 00 01 02 03 04
do
    unzip -o "$i.zip" > /dev/null
done

echo "Files extracted"

cd "$project_dir"

echo "Start preprocessing"

srun python source/preprocess_dataset.py --raw_data_directory="$raw_data_directory" --output_directory="$output_directory" --metadata_directory="$metadata_directory"

echo "done"
