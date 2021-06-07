import numpy as np
import pandas as pd
import itertools
import copy
from tqdm import tqdm
import scipy.stats
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
parser.add_argument('--model_k', type=int)
parser.add_argument('--run', type=int)

parser.add_argument('--num_hidden_layers', type=int, default=1)
parser.add_argument('--num_hidden_units', type=int, default=1000)
parser.add_argument('--learning_rate', type=float, default=1e-3)
args = parser.parse_args()
print(args)

run_idx_length = 10_000

if args.va_split_name is None:
    args.va_split_name = args.split

gamma = 0.99
nS, nA = 1442, 8
d = 21
num_epoch = 50

NSTEPS = 20
PROB_DIAB = 0.2
DISCOUNT = 1
USE_BOOSTRAP=True
N_BOOTSTRAP = 100

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

print('Loading data ... ', end='')
df_features_tr = pd.read_csv('{}/1-features.csv'.format(args.input_dir))
df_features_va = pd.read_csv('{}/2-features.csv'.format(args.input_dir))

def load_sparse_features(fname):
    feat_dict = joblib.load('{}/{}'.format(args.input_dir, fname))
    INDS_init, X, A, X_next, R = feat_dict['inds_init'], feat_dict['X'], feat_dict['A'], feat_dict['X_next'], feat_dict['R']
    return INDS_init, X.toarray(), A, X_next.toarray(), R

if args.split == 'tr':
    trINDS_init, trX, trA, trX_next, trR = load_sparse_features('1-21d-feature-matrices.sparse.joblib')
    first_ind = trINDS_init[args.run*run_idx_length]
    last_ind = trINDS_init[args.run*run_idx_length+args.N]
    X, A, X_next, R = trX[first_ind:last_ind], trA[first_ind:last_ind], trX_next[first_ind:last_ind], trR[first_ind:last_ind]
elif args.split == 'va':
    vaINDS_init, vaX, vaA, vaX_next, vaR = load_sparse_features('2-21d-feature-matrices.sparse.joblib')
    first_ind = vaINDS_init[args.run*run_idx_length]
    last_ind = vaINDS_init[args.run*run_idx_length+args.N]
    X, A, X_next, R = vaX[first_ind:last_ind], vaA[first_ind:last_ind], vaX_next[first_ind:last_ind], vaR[first_ind:last_ind]
else:
    assert False

print('DONE')
print()


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import tensorflow as tf
from tensorflow import keras
from tf_utils import select_output

def clone_keras_model(model):
    model_copy = keras.models.clone_model(model)
    model_copy.set_weights(model.get_weights())
    return model_copy

def init_networks():
    # Inputs
    state_input = keras.Input(shape=(d), name='state_input')
    action_input = keras.Input(shape=(), dtype=tf.int32, name='action_input')
    
    # Layers
    hidden_layers = keras.Sequential([
        keras.layers.Dense(1000, activation="relu"),
        keras.layers.Dense(nA),
    ], name='hidden_layers')
    
    action_selection_layer = keras.layers.Lambda(select_output, name='action_selection')
    
    # Outputs
    hidden_output = hidden_layers(state_input)
    Q_output = action_selection_layer([hidden_output, action_input])
    
    # Models
    hidden_net = keras.Model(inputs=[state_input], outputs=[hidden_output], name='hidden_net')
    Q_net = keras.Model(inputs=[state_input, action_input], outputs=[Q_output], name='Q_net')
    Q_net.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError(),
        metrics=keras.metrics.MeanSquaredError(),
    )
    return hidden_net, Q_net

fit_args = dict(
    batch_size=64, 
    validation_split=0.1, 
    epochs=100, 
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10, restore_best_weights=True)]
)


print('FQE')

import pathlib
save_dir = '{}/NFQ-clipped-keras.{}FQE_models/nl={},nh={},lr={}/'.format(args.output_dir, args.va_split_name, args.num_hidden_layers, args.num_hidden_units, args.learning_rate)
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

def run_FQE(X, A, X_next, R, model_k, n_epoch=10):
    model = keras.models.load_model('{}/NFQ-clipped-keras.models.nl={},nh={},lr={}/iter={}.hidden_net'.format(args.output_dir, args.num_hidden_layers, args.num_hidden_units, args.learning_rate, model_k))
    N = len(X)
    next_actions = model.predict(X_next).argmax(axis=1)
    
    np.random.seed(0)
    python_random.seed(0)
    tf.random.set_seed(0)
    
    hidden_net, Q_net = init_networks()
    Q_net.fit([X,A], np.zeros_like(R), **fit_args)
    for _ in range(n_epoch):
        y = R + gamma * hidden_net.predict(X_next)[range(N), next_actions]
        y = np.clip(y, -1, 1)
        tf.random.set_seed(0)
        hidden_net, Q_net = init_networks()
        Q_net.fit([X, A], y, **fit_args)
    
    Q_net.save('{}/model={}.Q_net'.format(save_dir, model_k))
    hidden_net.save('{}/model={}.hidden_net'.format(save_dir, model_k))
    return (hidden_net, Q_net)

model_k = args.model_k
# for model_k in [1, 2, 4, 8, 16, 32]:
try:
    Q_net = keras.models.load_model('{}/model={}.Q_net'.format(save_dir, model_k))
except:
    run_FQE(X, A, X_next, R, model_k, 20)
