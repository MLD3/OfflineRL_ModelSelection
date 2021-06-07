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
df_va1 = load_data('2-features.csv').set_index('pt_id').loc[(200_000+run*run_idx_length):(200_000+run*run_idx_length+N//2-1)].reset_index()
vaINDS_init, vaX, vaA, vaX_next, vaR = load_sparse_features('../unif-100k/2-21d-feature-matrices.sparse.joblib')
first_ind = vaINDS_init[run*run_idx_length]
last_ind = vaINDS_init[run*run_idx_length+N//2]
X1, A1, X_next1, R_1 = vaX[first_ind:last_ind], vaA[first_ind:last_ind], vaX_next[first_ind:last_ind], vaR[first_ind:last_ind]
INDS_init1 = vaINDS_init[run*run_idx_length:run*run_idx_length+N//2]
X_init1 = vaX[INDS_init1]
INDS_init1 -= INDS_init1[0]

df_va2 = load_data('../eps_0_1-100k/2-features.csv').set_index('pt_id').loc[(200_000+run*run_idx_length):(200_000+run*run_idx_length+N//2-1)].reset_index()
vaINDS_init, vaX, vaA, vaX_next, vaR = load_sparse_features('../eps_0_1-100k/2-21d-feature-matrices.sparse.joblib')
first_ind = vaINDS_init[run*run_idx_length]
last_ind = vaINDS_init[run*run_idx_length+N//2]
X2, A2, X_next2, R_2 = vaX[first_ind:last_ind], vaA[first_ind:last_ind], vaX_next[first_ind:last_ind], vaR[first_ind:last_ind]
INDS_init2 = vaINDS_init[run*run_idx_length:run*run_idx_length+N//2]
X_init2 = vaX[INDS_init2]
INDS_init2 -= INDS_init2[0]

df_va2['pt_id'] = df_va2['pt_id'] + 1_000_000
df_va = pd.concat([df_va1, df_va2])

X = np.vstack([X1, X2])
A = np.concatenate([A1, A2])
R_ = np.concatenate([R_1, R_2])
X_next = np.vstack([X_next1, X_next2])
X_init = np.vstack([X_init1, X_init2])
INDS_init = np.concatenate([INDS_init1, INDS_init2 + len(X1)])



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import tensorflow as tf
from tensorflow import keras
from tf_utils import select_output_d, select_output
from OPE_utils_keras import *

# ## Behavior policy: predict action probabilities
print('behavior_net')
behavior_net = keras.models.load_model('./output/run{}/unif-10k/{}.behavior_net'.format(run, args.va_split_name), compile=False)
behavior_net.summary()


# ## Env dynamics model: predict change in state features
print('dynamics.delta_net')
delta_net = keras.models.load_model('./output/run{}/unif-10k/{}.dynamics.delta_net'.format(run, args.va_split_name), compile=False, custom_objects={'tf': tf, 'select_output_d': select_output_d})
delta_net.summary()


# ## Env dynamics model: predict reward & termination
print('dynamics.reward_net')
reward_net = keras.models.load_model('./output/run{}/unif-10k/{}.dynamics.reward_net'.format(run, args.va_split_name), compile=False, custom_objects={'select_output_d': select_output_d})
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

FQI_value_list = [hidden_net.predict(X_init).max(axis=1).mean() for hidden_net in hidden_nets]
true_value_list = [isd @ policy_eval_analytic(P.transpose((1,0,2)), R, π, gamma) for π in π_list]
true_value_dict = dict(zip(keys, true_value_list))


print('WIS')
features_tensor = format_features_tensor(df_va, X, INDS_init)
WIS_value_list, WIS_N_list, WIS_ESS_list = zip(*Parallel(n_jobs=18)(delayed(OPE_WIS_keras)(
    features_tensor, k, gamma, output_dir, 
    save_dir='{}/NFQ-clipped-keras.models.nl={},nh={},lr={}/'.format(args.FQI_output_dir, nl, nh, lr), 
    split=va_split_name) for nl, nh, lr, k in tqdm(itertools.product(layers_list, units_list, lr_list, k_list))))


print('AM')
AM_value_list_both = Parallel(n_jobs=18)(delayed(OPE_AM_keras)(
    k, X_init, gamma, output_dir, split=va_split_name, 
    save_dir='{}/NFQ-clipped-keras.models.nl={},nh={},lr={}/'.format(args.FQI_output_dir, nl, nh, lr)
) for nl, nh, lr, k in tqdm(itertools.product(layers_list, units_list, lr_list, k_list)))
_, AM_value_list = zip(*AM_value_list_both)


print('FQE')
V_X_init = Parallel(n_jobs=18)(delayed(get_FQE_value_keras)(
    k, X_init, output_dir, split=va_split_name, 
    save_dir='{}/NFQ-clipped-keras.{}FQE_models/nl={},nh={},lr={}/'.format(output_dir, args.va_split_name, nl, nh, lr)
) for nl, nh, lr, k in tqdm(itertools.product(layers_list, units_list, lr_list, k_list)))
FQE_value_list = [np.nanmean(V_init) for V_init in V_X_init]


print('WDR-FQE')
WDR_FQE_value_list = Parallel(n_jobs=18)(delayed(OPE_WDR_FQE_keras)(
    features_tensor, k, gamma, output_dir, split=va_split_name,
    FQE_save_dir='{}/NFQ-clipped-keras.{}FQE_models/nl={},nh={},lr={}/'.format(output_dir, args.va_split_name, nl, nh, lr),
    net_save_dir='{}/NFQ-clipped-keras.models.nl={},nh={},lr={}/'.format(args.FQI_output_dir, nl, nh, lr), 
) for nl, nh, lr, k in tqdm(itertools.product(layers_list, units_list, lr_list, k_list)))


print('Plot & Save')
df_all_values = pd.DataFrame({
    '(nl, nh, lr, k)': keys, 
    'true_value_list': true_value_list,
    'FQI_value_list': FQI_value_list,
    'WIS_value_list': WIS_value_list,
    'AM_value_list': AM_value_list,
    'FQE_value_list': FQE_value_list,
    'WDR_FQE_value_list': WDR_FQE_value_list,
})
df_all_values.to_csv('./results/run{}/sepsis-cont-HP-{}.values.csv'.format(run, args.va_split_name), index=False)

print(np.max(true_value_list))
WDR_AM_value_list = np.full_like(FQI_value_list, fill_value=np.nan)
results = []
for value_list in [
    FQI_value_list, 
    WIS_value_list,
    AM_value_list,
    WDR_AM_value_list,
    FQE_value_list,
    WDR_FQE_value_list,
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
    index=['FQI', 'WIS', 'AM', 'WDR-AM', 'FQE', 'WDR-FQE']
).T
df_results.to_csv('./results/run{}/sepsis-cont-HP-{}.csv'.format(run, args.va_split_name))
