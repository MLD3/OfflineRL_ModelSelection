#!/bin/bash


#SBATCH --job-name=mainFQE
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
layers=$4
units=$5
lr=$6
k=$7

python run-NFQE-clipped-keras-split-k.py --input_dir='../datagen/unif-100k/' --output_dir="./output/run$run/unif-10k/" --N=10000 --split='va' --run=$run --num_hidden_layers=$layers --num_hidden_units=$units --learning_rate=$lr  --model_k=$k;