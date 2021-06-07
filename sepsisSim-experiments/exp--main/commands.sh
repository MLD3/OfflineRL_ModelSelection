set -exuo pipefail

mkdir -p "./fig/"

for run in {0..9}; 
do mkdir -p "./output/run$run/unif-10k/"; mkdir -p "./results/run$run/"; 
done

# FQI
for run in {0..9}; do for layers in 1 2; do for units in 100 200 500 1000; do for lr in '1e-3' '1e-4'; 
do sbatch job-NFQ.sh _ _ $run $layers $units $lr;
done; done; done; done

# FQE
for run in {0..9}; do for layers in 1 2; do for units in 100 200 500 1000; do for lr in '1e-3' '1e-4'; do for k in 1 2 4 8 16 32; 
do sbatch job-NFQE.sh _ _ $run $layers $units $lr $k;
done; done; done; done; done

# WIS, AM
for run in {0..9}; do sbatch job-WIS-AM.sh _ _ $run; done

# Final OPE computations
for run in {0..9}; do sbatch job-OPE.sh _ _ $run; done
for run in {0..9}; do sbatch job-OPE-losses.sh _ _ $run; done


exit


#### Example of running the python scripts directly

run=0
mkdir -p "./output/run$run/unif-10k/"

python run-NFQ-clipped-keras.py --input_dir='../datagen/unif-100k/' --output_dir="./output/run$run/unif-10k/" --N=10000 --run=$run --num_hidden_layers=1 --num_hidden_units=100 --learning_rate='1e-3'
python run-NFQE-clipped-keras-split-k.py --input_dir='../datagen/unif-100k/' --output_dir="./output/run$run/unif-10k/" --N=10000 --split='va' --run=$run --num_hidden_layers=1 --num_hidden_units=100 --learning_rate='1e-3' --model_k=1
python run-WIS-AM-models.py --input_dir='../datagen/unif-100k/' --output_dir="./output/run$run/unif-10k/" --N=10000 --split='va' --run=$run
python run-OPE.py --input_dir='../datagen/unif-100k/' --output_dir="./output/run$run/unif-10k/" --N=10000 --split='va' --run=$run
