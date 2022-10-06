# Offline RL Model Selection: Practical Considerations for Healthcare

This repository contains the source code for replicating all experiments in the MLHC 2021 paper, "Model Selection for Offline RL: Practical Considerations for Healthcare Settings" by Shengpu Tang & Jenna Wiens. 

Repository content: 
- `OPE_impl` contains a copy of OPE estimator implementations (WIS, AM, FQE, WDR) for both the tabular and function approximation settings. 
- `sepsisSim-experiments` contains code to replicate the main experiments. The sepsis simulator code is based on https://github.com/clinicalml/gumbel-max-scm/tree/sim-v2/sepsisSimDiabetes. 

If you use this code in your research, please cite the following publication:
```
@inproceedings{tang2021offline,
    author={Tang, Shengpu and Wiens, Jenna},
    title={Model Selection for Offline Reinforcement Learning: Practical Considerations for Healthcare Settings},
    booktitle={Machine Learning for Healthcare Conference},
    pages={2--35},
    year={2021}
}
```

## Dependencies
This code was run using python 3.8 in a conda environment. The dependency specification is provided in `environment.yml` (with `environment-full.yml` containing the exact versions of all packages used on a ubuntu-based cluster). Use `conda env create -f environment.yml` to recreate the conda environment. 

## Usage Notes
The folder `sepsisSim-experiments` includes code to produce figures used in the paper (and appendix) from scratch. Alternatively, you can find an archive containing all outputs here (total file size ~30GB): [link](https://www.dropbox.com/sh/g7ipt8v2jebr41n/AAA2hF3YesHeOWIt4kfUrqgra?dl=0). 
- The preparation steps are in `data-prep`, which include the simulator source code as well as several notebooks for dataset generation. The output is saved to `data` (ground-truth MDP parameters, ground-truth optimal policy, and optimal value functions) and `datagen` (generated datasets). This may take up to 3 hours.  
- The experiments can be found in the following folders:
    - `exp--main` (Sec 5.1, Appx D.1): model selection of neural architectures and training hyperparameters, using WIS/AM/FQE/WDR, 2-stage WIS+FQE, and FQI/RMS-TDE. 
    - `exp-auxHP` (Sec 5.2.1): sensitivity to OPE auxiliary hyperparameters
    - `exp-vasize` (Sec 5.2.2): sensitivity to validation dataset size
    - `exp-beh` (Sec 5.2.3): sensitivity to behavior policy used to collect validation data
    - `exp-2stage-FINAL` (Sec 5.2): additional comparison with the 2-stage selection procedure
    - `exp_earlystopping-tabular` and `exp_earlystopping-func` (Appx D.2): additional experiments where the candidate policy set is from the training path of an FQI run and the model selection problem is determining the training iteration for early stopping (applicable for both tabular and function approximation settings)


## Additional Information

### Running the experiments
The experiments need to be run in the order specified above (same as paper section order) because some models are saved and reused in later experiments. In general, within each `exp-*` subfolder, `commands.sh` specifies the sequence of `job-*.sh` bash scripts for training and evaluating policies. We used a HPC cluster with the Slurm scheduler to run these `job-*.sh` in parallel; alternatively, all `job-*.sh` can be run as regular bash scripts and they make use of the corresponding`run-*.py` python scripts. (Note: the `exp_earlystopping-*` folders contain notebooks instead.) 
- `exp--main` saves all FQI and WIS/AM/FQE models. 
- `exp-auxHP` reuses the saved WIS/AM models from `exp--main` but retrains all FQE models with varying evaluation horizons. 
- `exp-vasize` and `exp-beh` retrains all OPE models because the validation data is different in each case.

After saving all output, you can use the notebooks in each subdirectory to generate figures. 

### Simulator and dataset generation
- We compute the exact MDP parameters (instead of approximating it using data as was done in https://github.com/clinicalml/gumbel-max-scm) and save it as `data/MDP_parameters.joblib`. 
- The MDP has 8 discrete actions from combinations of 3 binary treatments. There are 1,440 states from combinations of 8 state variables, and 2 additional absorbing states representing death and discharge (i.e., survival). Among the 1,440 states, 832 are "almost dying" and deterministically leads to the death absorbing state, 2 are "ready for discharge" and deterministically leads to the discharge absorbing state, and 606 non-terminating states that do not transition to death/discharge. Transitions among the non-terminating states and from non-terminating states to terminating states all depend on the actions and are stochastic. Reward of -1/+1 (for death/discharge respectively) is assigned at the transition from the terminating state to the corresponding absorbing state. 
- To reduce the negative impact of insufficient coverage for rare states/actions on learning good policies using FQI, we use a modified initial state distribution that is uniformly random over all non-terminating states (including those with treatments). 
- We consider two behavior policies: a uniformly random behavior policy, and a near-optimal ε-greedy behavior policy with ε=0.10. 
- For each behavior policy, we simulate 100,000 episodes for training and for validation, each with a different starting seed (1 and 2 respectively). These data are then treated as 10 pairs of training (10,000) / validation (10,000) data for 10 replication runs of all experiments. 
- Dataset generation takes ~1h for each policy (simulating trajectories ~45 min, converting to feature matrices ~15 min). 

### OPE implementations
- Tabular implementation is in `OPE_utils.py`
- Function approximator implementation (using `tf.keras`) is in: `tf_utils.py` and `OPE_utils_keras.py`
- A example notebook is provided  (TODO)

### Keras models
- All neural network models are implemented using tensorflow 2 and the keras interface. Models are trained with the following settings by default:
    ```
    hidden_size=1000
    fit_args = dict(
        batch_size=64, 
        validation_split=0.1, 
        epochs=100, 
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10, restore_best_weights=True)],
    )
    ```
