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
parser.add_argument('--run', type=int)

args = parser.parse_args()
print(args)

run_idx_length = 10_000

if args.va_split_name is None:
    args.va_split_name = args.split

def load_sparse_features(fname):
    feat_dict = joblib.load('{}/{}'.format(args.input_dir, fname))
    INDS_init, X, A, X_next, R = feat_dict['inds_init'], feat_dict['X'], feat_dict['A'], feat_dict['X_next'], feat_dict['R']
    return INDS_init, X.toarray(), A, X_next.toarray(), R


print('Loading data ... ', end='')
if args.split == 'tr':
    trINDS_init, trX, trA, trX_next, trR = load_sparse_features('1-21d-feature-matrices.sparse.joblib')
    first_ind = trINDS_init[args.run*run_idx_length]
    last_ind = trINDS_init[args.run*run_idx_length+args.N]
    X, A, X_next, R_ = trX[first_ind:last_ind], trA[first_ind:last_ind], trX_next[first_ind:last_ind], trR[first_ind:last_ind]
elif args.split == 'va':
    vaINDS_init, vaX, vaA, vaX_next, vaR = load_sparse_features('2-21d-feature-matrices.sparse.joblib')
    first_ind = vaINDS_init[args.run*run_idx_length]
    last_ind = vaINDS_init[args.run*run_idx_length+args.N]
    X, A, X_next, R_ = vaX[first_ind:last_ind], vaA[first_ind:last_ind], vaX_next[first_ind:last_ind], vaR[first_ind:last_ind]
else:
    assert False

X_delta = X_next - X

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import tensorflow as tf
from tensorflow import keras
from tf_utils import select_output_d, select_output
from OPE_utils_keras import *

behavior_net = learn_behavior_net(X, A, args.output_dir, args.va_split_name)
delta_net = learn_dynamics_delta_net([X,A], X_delta, args.output_dir, args.va_split_name)
reward_net = learn_dynamics_reward_net(X, R_, args.output_dir, args.va_split_name)
