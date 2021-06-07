#!/bin/bash


#SBATCH --job-name=OPEvasize
#SBATCH --mail-user=tangsp@umich.edu 
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p standard
#SBATCH --time=144:00:00
#execute code


N=$1
name=$2
run=$3

python run-OPE.py --input_dir='../datagen/unif-100k/' --output_dir="./output/run$run/unif-10k/" --N=$N --split='va' --run=$run --va_split_name=$name --FQI_output_dir="../exp--main/output/run$run/unif-10k/"
