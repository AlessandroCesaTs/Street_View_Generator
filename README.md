# Street_View_Generator
This repository contains the code for a Conditional Variational Auto Encoder generator of Google Street View style images, made by Alessandro Cesa for the Deep Learning exam at University of Trieste.

## Structure of the repository

`create_dataset.sh` is the bash file used for downloading and pre-processing the dataset

`train_and_generate.sh` is the bash file used for training the model and generate images

`train.sh` is the bash file needed to train the model

`generate_images.sh` is the bash file needed to generate the images

`source` contains all the python code needed, directly in the folder you can find the programs the are directly executed, while in `functions` you can find all the functions needed

`requirements.txt` Contains all the requirements needed for the project

`report.pdf` is the report of the project

`presentation.pdf` contains the slides for the presentation

`images` contains some generated images, as well as an image from the dataset (`input_image.png`) and its output from the Conditional VAE

`losses` Contains the file CSV file monitoring the losses during the training

`models` Contains the trained model

`plots` Contains the plots of the losses

## How to reproduce the project

At first, you should clone this repository

### Requirements and environment

The project was runned with Python 3.10.10

In order to run all the code, is advisable to create a python virtual environment with `python -m venv -n <environment>` (otherwise you'll have to adjust the .sh files)


In order to install the requirements you need to activate the environment and then run `pip install -r requirements.txt`

### SLURM

The scripts `.sh` scripts are intended to run on the cluster ORFEO of Trieste's Are Science park, which uses the batch scheduler SLURM. If you're running on a different machine, you'll have to adapt them (for example by changing the partition  names if you're still using SLURM but on a different cluster, or by deleting the lines starting by #SBATCH and using `bash` instead of `sbatch` if you're not running on SLURM)

### Download and pre process the dataset

In order to download and pre process the dataset, run 
`bash create_dataset.sh <directory_where_you_want_the_dataset>`

### Train the model and generate images 

In order to train the model and generate some images, you have to run 
`bash train_and_generate <directory where you have downloaded the dataset>` 
