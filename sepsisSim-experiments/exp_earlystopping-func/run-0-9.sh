set -exuo pipefail

# FQI tr
for run in {0..9};
do tmux new-session -d "python run-NFQ-clipped-keras.py --input_dir='../datagen/unif-100k/' --output_dir='./output/run$run/unif-10k/' --N=10000 --run=$run";
done;

# FQE 10k tr
for run in {0..9};
do tmux new-session -d "python run-NFQE-clipped-keras-split.py --input_dir='../datagen/unif-100k/' --output_dir='./output/run$run/unif-10k/' --N=10000 --split='tr' --k_start=0  --k_end=51 --run=$run";
done;

# FQE 10k va iterations
for run in {0..9};
do tmux new-session -d "python run-NFQE-clipped-keras-iterations.py --input_dir='../datagen/unif-100k/' --output_dir='./output/run$run/unif-10k/' --N=10000 --split='va' --k_start=0  --k_end=5 --run=$run; python run-NFQE-clipped-keras-iterations.py --input_dir='../datagen/unif-100k/' --output_dir='./output/run$run/unif-10k/' --N=10000 --split='va' --k_start=10  --k_end=51 --run=$run";
done;

# FQE 10k va5k/va1k
for run in {0..9};
do tmux new-session -d "python run-NFQE-clipped-keras-split.py --input_dir='../datagen/unif-100k/' --output_dir='./output/run$run/unif-10k/' --N=5000 --split='va' --k_start=0  --k_end=51 --va_split_name='va5k'  --run=$run; python run-NFQE-clipped-keras-split.py --input_dir='../datagen/unif-100k/' --output_dir='./output/run$run/unif-10k/' --N=1000 --split='va' --k_start=0  --k_end=51 --va_split_name='va1k'";
done;

# FQE 10k va500/va1k
for run in {0..9};
do tmux new-session -d "python run-NFQE-clipped-keras-split.py --input_dir='../datagen/unif-100k/' --output_dir='./output/run$run/unif-10k/' --N=500 --split='va' --k_start=0  --k_end=51 --va_split_name='va500'  --run=$run; python run-NFQE-clipped-keras-split.py --input_dir='../datagen/unif-100k/' --output_dir='./output/run$run/unif-10k/' --N=1000 --split='va' --k_start=0  --k_end=51 --va_split_name='va1k'";
done;

# FQE va 10k behavior eps_01/mixed
for run in {0..9};
do tmux new-session -d "python run-NFQE-clipped-keras-behavior.py --input_dir='../datagen/unif-100k/' --output_dir='./output/run$run/unif-10k/' --N=10000 --split='va' --k_start=0  --k_end=51 --va_split_name='va10k_eps01' --run=$run; python run-NFQE-clipped-keras-behavior-mixed.py --input_dir='../datagen/unif-100k/' --output_dir='./output/run$run/unif-10k/' --N=10000 --split='va' --k_start=0  --k_end=51 --va_split_name='va10k_mixed' --run=$run";
done;

# FQI trva
for run in {0..9};
do tmux new-session -d "python run-NFQ-clipped-keras-trva.py --input_dir='../datagen/unif-100k/' --output_dir='./output/run$run/unif-10k-trva/' --N=10000 --run=$run";
done;

# FQE trva
for run in {0..9}; 
do tmux new-session -d "python run-NFQE-clipped-keras-trva.py --input_dir='../datagen/unif-100k/' --output_dir='./output/run$run/unif-10k-trva/' --N=10000 --va_split_name='trva' --k_start=0  --k_end=51 --run=$run";
done;
