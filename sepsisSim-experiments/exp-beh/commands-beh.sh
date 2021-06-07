set -exuo pipefail

mkdir -p "./fig/"

for run in {0..9}; 
do mkdir -p "./output/run$run/unif-10k/"; mkdir -p "./results/run$run/"; 
done

# FQE with different behavior
for run in {0..9}; do for layers in 1 2; do for units in 100 200 500 1000; do for lr in '1e-3' '1e-4'; do for k in 1 2 4 8 16 32; 
do sbatch job-NFQE-beh1.sh _ 'va10k_eps01' $run $layers $units $lr $k; sbatch job-NFQE-beh2.sh _ 'va10k_mixed' $run $layers $units $lr $k;
done; done; done; done; done

# WIS, AM with different behavior
for run in {0..9}; 
do sbatch job-WIS-AM-beh1.sh _ 'va10k_eps01' $run; sbatch job-WIS-AM-beh2.sh _ 'va10k_mixed' $run; 
done

# Final OPE computations
for run in {0..9}; 
do sbatch job-OPE-beh1.sh _ 'va10k_eps01' $run; sbatch job-OPE-beh2.sh _ 'va10k_mixed' $run; 
done

exit
