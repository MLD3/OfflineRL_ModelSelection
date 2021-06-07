set -exuo pipefail

mkdir -p "./fig/"

for run in {0..9}; 
do mkdir -p "./output/run$run/unif-10k/"; mkdir -p "./results/run$run/"; 
done

# FQE with different validation data sizes
for run in {0..9}; do for layers in 1 2; do for units in 100 200 500 1000; do for lr in '1e-3' '1e-4'; do for k in 1 2 4 8 16 32; 
do sbatch job-NFQE-vasize.sh 500 "va500" $run $layers $units $lr $k; sbatch job-NFQE-vasize.sh 1000 "va1k" $run $layers $units $lr $k; sbatch job-NFQE-vasize.sh 5000 "va5k" $run $layers $units $lr $k;
done; done; done; done; done

# WIS, AM with different validation data sizes
for run in {0..9}; 
do sbatch job-WIS-AM-vasize.sh 500 "va500" $run; sbatch job-WIS-AM-vasize.sh 1000 "va1k" $run; sbatch job-WIS-AM-vasize.sh 5000 "va5k" $run; 
done

# Final OPE computations
for run in {0..9}; 
do sbatch job-OPE-vasize.sh 500 "va500" $run; sbatch job-OPE-vasize.sh 1000 "va1k" $run; sbatch job-OPE-vasize.sh 5000 "va5k" $run; 
done

exit
