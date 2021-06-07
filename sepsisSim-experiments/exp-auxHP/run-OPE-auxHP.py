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
parser.add_argument('--FQI_output_dir', type=str, required=False)
parser.add_argument('--N', type=int)
parser.add_argument('--split', type=str)
parser.add_argument('--va_split_name', type=str, required=False)
parser.add_argument('--run', type=int)

args = parser.parse_args()
print(args)

if args.va_split_name is None:
    args.va_split_name = args.split

if args.FQI_output_dir is None:
    args.FQI_output_dir = args.output_dir

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

# ## Behavior policy: predict action probabilities
print('behavior_net')
behavior_net = keras.models.load_model('{}/va.behavior_net'.format(args.FQI_output_dir, run), compile=False)
behavior_net.save('{}/va.behavior_net'.format(output_dir, run))
behavior_net.summary()


# ## Env dynamics model: predict change in state features
print('dynamics.delta_net')
delta_net = keras.models.load_model('{}/va.dynamics.delta_net'.format(args.FQI_output_dir, run), compile=False, custom_objects={'tf': tf, 'select_output_d': select_output_d})
delta_net.save('{}/va.dynamics.delta_net'.format(output_dir, run))
delta_net.summary()


# ## Env dynamics model: predict reward & termination
print('dynamics.reward_net')
reward_net = keras.models.load_model('{}/va.dynamics.reward_net'.format(args.FQI_output_dir, run), compile=False, custom_objects={'select_output_d': select_output_d})
reward_net.save('{}/va.dynamics.reward_net'.format(output_dir, run))
reward_net.summary()


print('Load FQI models')
layers_list = [1, 2]
units_list = [100, 200, 500, 1000]
lr_list = [1e-3, 1e-4]
k_list = [1, 2, 4, 8, 16, 32]
keys_list = list(itertools.product(layers_list, units_list, lr_list, k_list))

Q_nets_dict = {}
hidden_nets_dict = {}
for nl, nh, lr, k in keys_list:
    try:
        save_dir = '{}/NFQ-clipped-keras.models.nl={},nh={},lr={}/'.format(args.FQI_output_dir, nl, nh, lr)
        Q_net = keras.models.load_model('{}/iter={}.Q_net'.format(save_dir, k), compile=False, custom_objects={'select_output': select_output})
        hidden_net = keras.models.load_model('{}/iter={}.hidden_net'.format(save_dir, k), compile=False)
        Q_nets_dict[nl, nh, lr, k] = Q_net
        hidden_nets_dict[nl, nh, lr, k] = hidden_net
    except:
        continue

assert(len(Q_nets_dict) == len(keys_list))


print('Ground-truth performance')

# Make features for state-action pairs
X_ALL_states = []
for arrays in itertools.product(
    [[1,0], [0,1]], # Diabetic
    [[1,0,0], [0,1,0], [0,0,1]], # Heart Rate
    [[1,0,0], [0,1,0], [0,0,1]], # SysBP
    [[1,0], [0,1]], # Percent O2
    [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]], # Glucose
    [[1,0], [0,1]], # Treat: AbX
    [[1,0], [0,1]], # Treat: Vaso
    [[1,0], [0,1]], # Treat: Vent
):
    X_ALL_states.append(np.concatenate(arrays))

X_ALL_states = np.array(X_ALL_states)
X_ALL_states.shape

keys, hidden_nets = zip(*list(hidden_nets_dict.items()))

Qs = [hidden_net.predict(X_ALL_states) for hidden_net in hidden_nets]
π_list = [convert_to_policy_table(Q) for Q in Qs]

# FQI_value_list = [hidden_net.predict(X_init).max(axis=1).mean() for hidden_net in hidden_nets]
# true_value_list = [isd @ policy_eval_analytic(P.transpose((1,0,2)), R, π, gamma) for π in π_list]
# true_value_dict = dict(zip(keys, true_value_list))

df_all_values = pd.read_csv('../exp--main/results/run{}/sepsis-cont-HP-va.values.csv'.format(run))
true_value_list = list(df_all_values.set_index('(nl, nh, lr, k)').loc[[str(key) for key in keys]]['true_value_list'].values)
FQI_value_list = list(df_all_values.set_index('(nl, nh, lr, k)').loc[[str(key) for key in keys]]['FQI_value_list'].values)
true_value_dict = dict(zip(keys, true_value_list))



print('FQE')
def get_FQE_value_keras_iterations(k, X, it, output_dir, split, save_dir=None):
    try:
        model_FQE = keras.models.load_model('{}/iters/model={}_iter={}.hidden_net'.format(save_dir, k, it), custom_objects={'select_output': select_output}, compile=False)
        return model_FQE.predict(X).max(axis=1)
    except:
        return []

L_list = [1, 5, 10, 20, 50]
FQE_value_lists = []
for L in L_list:
    V_X_init = Parallel(n_jobs=8)(delayed(get_FQE_value_keras_iterations)(
        k, X_init, L, output_dir, split=va_split_name, 
        save_dir='{}/NFQ-clipped-keras.vaFQE_models/nl={},nh={},lr={}/'.format(output_dir, nl, nh, lr)
    ) for nl, nh, lr, k in tqdm(itertools.product(layers_list, units_list, lr_list, k_list)))
    FQE_value_list = [np.nanmean(V_init) for V_init in V_X_init]
    FQE_value_lists.append(FQE_value_list)

results = []
for L, value_list in zip(L_list, FQE_value_lists):
    if np.isnan(value_list).any():
        results.append([None, None, None, None, None])
        continue
    rho, pval = scipy.stats.spearmanr(true_value_list, value_list)
    mse = metrics.mean_squared_error(true_value_list, value_list)
    perf = true_value_list[np.argmax(value_list)]
    results.append([mse, rho, perf, np.max(true_value_list) - perf, J_star - perf])

df_results = pd.DataFrame(
    results, columns=['MSE', "Spearman", 'Performance', 'Regret', 'Suboptimality'], 
    index=['FQE (L={})'.format(L) for L in L_list]
).T
df_results.to_csv('./results/run{}/sepsis-cont-FQE_L.csv'.format(run))


print('WIS')
eps_list = [0, 0.01, 0.05, 0.10, 0.50, 1]
features_tensor = format_features_tensor(df_va, X, INDS_init)

WIS_value_lists = []
for eps in eps_list:
    WIS_value_list, WIS_N_list, WIS_ESS_list = zip(*Parallel(n_jobs=8)(delayed(OPE_WIS_keras)(
        features_tensor, k, gamma, output_dir, epsilon=eps, split=va_split_name,
        save_dir='{}/NFQ-clipped-keras.models.nl={},nh={},lr={}/'.format(args.FQI_output_dir, nl, nh, lr), 
    ) for nl, nh, lr, k in tqdm(itertools.product(layers_list, units_list, lr_list, k_list))))
    WIS_value_lists.append(WIS_value_list)

joblib.dump(WIS_value_lists, './results/run{}/sepsis-cont-WIS_eps.values.joblib'.format(run))

results = []
for eps, value_list in zip(eps_list, WIS_value_lists):
    if np.isnan(value_list).any():
        results.append([None, None, None, None])
        continue
    rho, pval = scipy.stats.spearmanr(true_value_list, value_list)
    mse = metrics.mean_squared_error(true_value_list, value_list)
    perf = true_value_list[np.argmax(value_list)]
    results.append([mse, rho, perf, np.max(true_value_list) - perf, J_star - perf])

df_results = pd.DataFrame(
    results, columns=['MSE', "Spearman", 'Performance', 'Regret', 'Suboptimality'], 
    index=['WIS (ε={})'.format(eps) for eps in eps_list]
).T
df_results.to_csv('./results/run{}/sepsis-cont-WIS_eps.csv'.format(run))


print('AM')
L_list = [1, 5, 10, 20, 50]
AM_value_lists = []
for L in L_list:
    AM_value_list_both = Parallel(n_jobs=8)(delayed(OPE_AM_keras)(
        k, X_init, gamma, output_dir, rollout=L, split=va_split_name, 
        save_dir='{}/NFQ-clipped-keras.models.nl={},nh={},lr={}/'.format(args.FQI_output_dir, nl, nh, lr)
    ) for nl, nh, lr, k in tqdm(itertools.product(layers_list, units_list, lr_list, k_list)))
    _, AM_value_list = zip(*AM_value_list_both)
    AM_value_lists.append(AM_value_list)

joblib.dump(AM_value_lists, './results/run{}/sepsis-cont-AM_L.values.joblib'.format(run))

results = []
for L, value_list in zip(L_list, AM_value_lists):
    if np.isnan(value_list).any():
        results.append([None, None, None, None])
        continue
    rho, pval = scipy.stats.spearmanr(true_value_list, value_list)
    mse = metrics.mean_squared_error(true_value_list, value_list)
    perf = true_value_list[np.argmax(value_list)]
    results.append([mse, rho, perf, np.max(true_value_list) - perf, J_star - perf])

df_results = pd.DataFrame(
    results, columns=['MSE', "Spearman", 'Performance', 'Regret', 'Suboptimality'], 
    index=['AM (L={})'.format(L) for L in L_list]
).T
df_results.to_csv('./results/run{}/sepsis-cont-AM_L.csv'.format(run))
