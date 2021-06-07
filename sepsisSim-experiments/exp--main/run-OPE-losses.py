import os
os.environ['JOBLIB_TEMP_FOLDER'] = '/scratch/wiensj_root/wiensj/tangsp/tmp'

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import pickle
import itertools
import copy
import random
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from sklearn import metrics
import random as python_random

import joblib
from joblib import Parallel, delayed


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--N', type=int)
parser.add_argument('--split', type=str)
parser.add_argument('--va_split_name', type=str, required=False)
parser.add_argument('--run', type=int)

args = parser.parse_args()
print(args)

if args.va_split_name is None:
    args.va_split_name = args.split

N = args.N
run = args.run
va_split_name = args.va_split_name
run_idx_length = 10_000

input_dir = '../datagen/unif-100k/'
output_dir = './output/run{}/unif-10k/'.format(run)

PROB_DIAB = 0.2
NSTEPS = 20     # max episode length
nS, nA = 1442, 8
d = 21

# Ground truth MDP model
MDP_parameters = joblib.load('../data/MDP_parameters.joblib')
P = MDP_parameters['transition_matrix_absorbing'] # (A, S, S_next)
R = MDP_parameters['reward_matrix_absorbing_SA'] # (S, A)
nS, nA = R.shape
gamma = 0.99

# unif rand isd, mixture of diabetic state
isd = joblib.load('../data/modified_prior_initial_state_absorbing.joblib')
isd = (isd > 0).astype(float)
isd[:720] = isd[:720] / isd[:720].sum() * (1-PROB_DIAB)
isd[720:] = isd[720:] / isd[720:].sum() * (PROB_DIAB)

# Optimal value function and optimal return
V_star = joblib.load('../data/V_π_star_PE.joblib')
J_star = V_star @ isd

def load_data(fname):
    print('Loading data', fname, '...', end='')
    df_data = pd.read_csv('{}/{}'.format(input_dir, fname)).rename(columns={'State_idx': 'State'})[['pt_id', 'Time', 'State', 'Action', 'Reward']]

    # Assign next state
    df_data['NextState'] = [*df_data['State'].iloc[1:].values, -1]
    df_data.loc[(df_data['Time'] == 19), 'NextState'] = -1
    df_data.loc[(df_data['Reward'] == -1), 'NextState'] = 1440
    df_data.loc[(df_data['Reward'] == 1), 'NextState'] = 1441

    assert ((df_data['Reward'] != 0) == (df_data['Action'] == -1)).all()

    print('DONE')
    return df_data

def load_sparse_features(fname):
    feat_dict = joblib.load('{}/{}'.format(input_dir, fname))
    INDS_init, X, A, X_next, R = feat_dict['inds_init'], feat_dict['X'], feat_dict['A'], feat_dict['X_next'], feat_dict['R']
    return INDS_init, X.toarray(), A, X_next.toarray(), R

print('Loading data ... ', end='')
if args.split == 'tr':
    df_tr = load_data('1-features.csv').set_index('pt_id').loc[(100_000+run*run_idx_length):(100_000+run*run_idx_length+N-1)].reset_index()
    trINDS_init, trX, trA, trX_next, trR = load_sparse_features('1-21d-feature-matrices.sparse.joblib')
    first_ind = trINDS_init[args.run*run_idx_length]
    last_ind = trINDS_init[args.run*run_idx_length+args.N]
    X, A, X_next, R_ = trX[first_ind:last_ind], trA[first_ind:last_ind], trX_next[first_ind:last_ind], trR[first_ind:last_ind]
    
    INDS_init = trINDS_init[run*run_idx_length:run*run_idx_length+N]
    X_init = trX[INDS_init]
    INDS_init -= INDS_init[0]
elif args.split == 'va':
    df_va = load_data('2-features.csv').set_index('pt_id').loc[(200_000+run*run_idx_length):(200_000+run*run_idx_length+N-1)].reset_index()
    vaINDS_init, vaX, vaA, vaX_next, vaR = load_sparse_features('2-21d-feature-matrices.sparse.joblib')
    first_ind = vaINDS_init[run*run_idx_length]
    last_ind = vaINDS_init[run*run_idx_length+N]
    X, A, X_next, R_ = vaX[first_ind:last_ind], vaA[first_ind:last_ind], vaX_next[first_ind:last_ind], vaR[first_ind:last_ind]
    
    INDS_init = vaINDS_init[run*run_idx_length:run*run_idx_length+N]
    X_init = vaX[INDS_init]
    INDS_init -= INDS_init[0]
else:
    assert False


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import tensorflow as tf
from tensorflow import keras
from tf_utils import select_output_d, select_output
from OPE_utils_keras import *

layers_list = [1, 2]
units_list = [100, 200, 500, 1000]
lr_list = [1e-3, 1e-4]
k_list = [1, 2, 4, 8, 16, 32]
keys_list = list(itertools.product(layers_list, units_list, lr_list, k_list))

print('Q-function output & FQI bootstrapping target')
def get_network_outputs(key, X, A, R, X_next):
    nl, nh, lr, k = key
    save_dir = '{}/NFQ-clipped-keras.models.nl={},nh={},lr={}/'.format(output_dir, nl, nh, lr)
    Q_net = keras.models.load_model('{}/iter={}.Q_net'.format(save_dir, k), compile=False, custom_objects={'select_output': select_output})
    hidden_net = keras.models.load_model('{}/iter={}.hidden_net'.format(save_dir, k), compile=False)
    return Q_net.predict([X, A]), R + gamma * hidden_net.predict(X_next).max(axis=1)

Q_values, target_values = zip(
    *Parallel(n_jobs=18)(delayed(get_network_outputs)((nl, nh, lr, k), X, A, R_, X_next) 
                        for nl, nh, lr, k in tqdm(keys_list)))


print('Losses')
td_errors = [metrics.mean_squared_error(Q_output, Q_target, squared=False) for Q_output, Q_target in zip(Q_values, target_values)]

print('Plot & Save')
D_all_losses = {
    '(nl, nh, lr, k)': keys_list, 
    'Q_values': Q_values, 
    'target_values': target_values,
    'td_errors': td_errors,
}
joblib.dump(D_all_losses, './results/run{}/sepsis-cont-HP-va.losses.joblib'.format(run))

df_all_values = pd.read_csv('./results/run{}/sepsis-cont-HP-va.values.csv'.format(run))
true_value_list = df_all_values['true_value_list']
FQI_value_list = df_all_values['FQI_value_list']

print(np.max(true_value_list))
results = []
for value_list in [
    FQI_value_list, 
    -np.array(td_errors),
]:
    if np.isnan(value_list).any():
        results.append([None, None, None, None])
        continue
    rho, pval = scipy.stats.spearmanr(true_value_list, value_list)
    mse = metrics.mean_squared_error(true_value_list, value_list)
    perf = true_value_list[np.argmax(value_list)]
    results.append([mse, rho, perf, np.max(true_value_list) - perf, J_star - perf])

df_results = pd.DataFrame(
    results, columns=['MSE', "Spearman", 'Performance', 'Regret', 'Suboptimality'], 
    index=['FQI', 'RMSE-TD']
).T
df_results.to_csv('./results/run{}/sepsis-cont-HP-va-loss.csv'.format(run))
