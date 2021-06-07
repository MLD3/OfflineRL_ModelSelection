set -exuo pipefail

mkdir -p "./fig/"

for run in {0..9}; 
do mkdir -p "./output/run$run/unif-10k/"; mkdir -p "./results/run$run/"; 
done

# FQE with varying evaluation horizon H
for run in {0..9}; do for layers in 1 2; do for units in 100 200 500 1000; do for lr in '1e-3' '1e-4'; do for k in 1 2 4 8 16 32; 
do sbatch job-NFQE-auxHP.sh _ _ $run $layers $units $lr $k;
done; done; done; done; done

# Final OPE computations
for run in {0..9}; do sbatch job-OPE-auxHP.sh _ _ $run; done

exit
