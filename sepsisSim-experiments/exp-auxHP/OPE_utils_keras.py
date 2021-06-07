import numpy as np
import tensorflow as tf
from tensorflow import keras
from tf_utils import select_output_d, select_output
import random as python_random

from tqdm import tqdm

NSTEPS = 20        # max episode length in historical data
MAX_ROLLOUT = 20   # model-based Monte-Carlo rollout
nS, nA = 1442, 8
d = 21             # dimension of state feature vector x(s)

##################
## Tabular func ##
##################

def convert_to_policy_table(Q):
    """
    Map the predicted Q-values (S,A) to a greedy policy (S,A) wrt tabular MDP
    Handles the last two states (absorbing states)
    """
    a_star = Q.argmax(axis=1)
    pol = np.zeros((nS, nA))
    pol[list(np.arange(nS-2)), a_star] = 1
    pol[-2:, 0] = 1
    return pol

def policy_eval_analytic(P, R, π, γ):
    """
    Given the MDP model transition probability P (S,A,S) and reward function R (S,A),
    Compute the value function of a stochastic policy π (S,A) using matrix inversion
    
        V_π = (I - γ P_π)^-1 R_π
    """
    nS, nA = R.shape
    R_π = np.sum(R * π, axis=1)
    P_π = np.sum(P * np.expand_dims(π, 2), axis=1)
    V_π = np.linalg.inv(np.eye(nS) - γ * P_π) @ R_π
    return V_π

##################
## Preparations ##
##################

def learn_behavior_net(X, A, output_dir, split='va'):
    np.random.seed(0)
    python_random.seed(0)
    tf.random.set_seed(0)
    
    behavior_net = keras.Sequential([
        keras.layers.Dense(1000, activation="relu", input_shape=(d,)),
        keras.layers.Dense(nA, activation='softmax'),
    ], name='behavior_net')

    behavior_net.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=keras.metrics.SparseCategoricalCrossentropy(),
    )
    
    fit_args = dict(
        batch_size=64, 
        validation_split=0.1, 
        epochs=100, 
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10, restore_best_weights=True)]
    )
    
    # Fit action on state features
    behavior_net.fit(X, A, **fit_args)
    
    behavior_net.save('{}/{}.behavior_net'.format(output_dir, split))
    return behavior_net

def learn_dynamics_delta_net(XA, X_delta, output_dir, split='va'):
    X, A = XA
    
    def init_networks():
        # Inputs
        state_input = keras.Input(shape=(d), name='state_input')
        action_input = keras.Input(shape=(), dtype=tf.int32, name='action_input')

        # Layers
        hidden_layers = keras.Sequential([
            keras.layers.Dense(1000, activation="relu"),
            keras.layers.Dense(nA*d),
        ], name='hidden_layers')

        action_selection_layer = keras.layers.Lambda(select_output_d, arguments={'d': d}, name='action_selection')

        # Outputs
        hidden_output = hidden_layers(state_input)
        delta_output = action_selection_layer([hidden_output, action_input])

        # Models
        hidden_net = keras.Model(inputs=[state_input], outputs=[hidden_output], name='hidden_net')
        delta_net = keras.Model(inputs=[state_input, action_input], outputs=[delta_output], name='delta_net')
        delta_net.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.MeanSquaredError(),
            metrics=keras.metrics.MeanSquaredError(),
        )
        return hidden_net, delta_net
    
    fit_args = dict(
        batch_size=64, 
        validation_split=0.1, 
        epochs=100, 
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10, restore_best_weights=True)]
    )
    
    np.random.seed(0)
    python_random.seed(0)
    tf.random.set_seed(0)
    
    hidden_net, delta_net = init_networks()
    
    delta_net.fit([X,A], X_delta, **fit_args)
    
    delta_net.save('{}/{}.dynamics.delta_net'.format(output_dir, split))
    
    return delta_net

def learn_dynamics_reward_net(X, R, output_dir, split='va'):
    np.random.seed(0)
    python_random.seed(0)
    tf.random.set_seed(0)
    
    reward_net = keras.Sequential([
        keras.layers.Dense(1000, activation="relu", input_shape=(d,)),
        keras.layers.Dense(1),
    ], name='reward_net')

    reward_net.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError(),
        metrics=keras.metrics.MeanSquaredError(),
    )
    
    fit_args = dict(
        batch_size=64, 
        validation_split=0.1, 
        epochs=100, 
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10, restore_best_weights=True)]
    )
    
    # Terminal states will get r(s,a) = +/-1 regardless of action and transition into absorbing state
    # Fit rewards on state features only
    reward_net.fit(X, R, **fit_args)
    
    reward_net.save('{}/{}.dynamics.reward_net'.format(output_dir, split))
    
    return reward_net

def format_features_tensor(df_data, X, inds_init):
    data_dict = dict(list(df_data.groupby('pt_id')))
    N = len(data_dict)
    data_tensor = np.zeros((N, NSTEPS, 3+d), dtype=float) # [t, a, r, x(s)]
    data_tensor[:, :, 1] = -1 # initialize all actions to -1
    data_tensor[:, :, 3:] = -1 # initialize all state features to -1

    for i, (pt_id, df_values) in tqdm(enumerate(data_dict.items())):
        ind_start = inds_init[i]
        if len(df_values) == NSTEPS and df_values['Reward'].values[-1] == 0:
            # ignore terminal transition if terminated early because we did not make features for it
            data_tensor[i, :len(df_values)-1, :3] = df_values[['Time', 'Action', 'Reward']].values[:-1]
            data_tensor[i, :len(df_values)-1, 3:] = X[ind_start:ind_start+len(df_values)-1]
        else:
            # terminated in death/disch state and obtained reward
            data_tensor[i, :len(df_values), :3] = df_values[['Time', 'Action', 'Reward']].values
            data_tensor[i, :len(df_values), 3:] = X[ind_start:ind_start+len(df_values)]
    return data_tensor

#########################
## Evaluating a policy ##
#########################

def policy_eval_analytic(P, R, pi, γ):
    """
    Given the MDP model (transition probability P (S,A,S) and reward function R (S,A)),
    Compute the value function of a policy using matrix inversion
    
        V_π = (I - γ P_π)^-1 R_π
    """
    nS, nA = R.shape
    R_pi = np.sum(R * pi, axis=1)
    P_pi = np.sum(P * np.expand_dims(pi, 2), axis=1)
    V_pi = np.linalg.inv(np.eye(nS) - γ * P_pi) @ R_pi
    return V_pi

def OPE_WIS_keras(data, model_k, γ, output_dir, save_dir=None, epsilon=0.01, split='va'):
    behavior_net = keras.models.load_model('{}/{}.behavior_net'.format(output_dir, split))
    behavior_probs = behavior_net.predict(data[:, :, -d:].reshape((-1, d))).reshape((-1, NSTEPS, nA))
    
    if save_dir is None:
        save_dir = '{}/NFQ-clipped-keras.models/'.format(output_dir)
    
    hidden_net = keras.models.load_model('{}/iter={}.hidden_net'.format(save_dir, model_k), compile=False)
    a_pred = hidden_net.predict(data[:, :, -d:].reshape((-1, d))).reshape((-1, NSTEPS, nA)).argmax(axis=-1)
    
    return wis_keras(data, behavior_probs, a_pred, γ, epsilon)

def wis_keras(data, probs_b, a_pred, γ, epsilon):
    """Weighted Importance Sampling for Off Policy Evaluation
        - data: tensor of shape (N, T, 5) with the last last dimension being [t, s, a, r, s']
        - π_b:  behavior policy
        - π_e:  evaluation policy (aka target policy)
        - γ:    discount factor
    """
    t_list = data[..., 0].astype(int)
    a_list = data[..., 1].astype(int)
    r_list = data[..., 2].astype(int)
    
    # Per-trajectory returns (discounted cumulative rewards)
    G = (r_list * np.power(γ, t_list)).sum(axis=-1)
    
    # Per-transition importance ratios
    p_b = probs_b.reshape((-1, nA))[range(len(probs_b)*NSTEPS), a_list.reshape((-1))].reshape((-1, NSTEPS))
    p_e = (a_pred == a_list) * (1 - epsilon) + (a_pred != a_list) * (epsilon / (nA - 1)) # Get a soft version of the evaluation policy for WIS

    # Deal with variable length sequences by setting ratio to 1
    terminated_idx = (a_list == -1)
    p_b[terminated_idx] = 1
    p_e[terminated_idx] = 1
    
    if not np.all(p_b > 0):
        import pdb
        pdb.set_trace()
    assert np.all(p_b > 0), "Some actions had zero prob under p_b, WIS fails"

    # Per-trajectory cumulative importance ratios, take the product
    rho = (p_e / p_b).prod(axis=1)

    # directly calculate weighted average over trajectories
    wis_value = np.average(G, weights=rho)
    
    # Effective sample size
    ess = ess = (rho.sum()**2) / (np.square(rho)).sum()
    
    return wis_value, (rho > 0).sum(), ess

def get_FQE_value_keras(k, X, output_dir, split, save_dir=None):
    try:
        if save_dir is None:
            save_dir = '{}/NFQ-clipped-keras.{}FQE_models/'.format(output_dir, split)

        model_FQE = keras.models.load_model('{}/model={}.hidden_net'.format(save_dir, k), custom_objects={'select_output': select_output}, compile=False)
        return model_FQE.predict(X).max(axis=1)
    except:
        return []

def OPE_WDR_FQE_keras(data, model_k, γ, output_dir, epsilon=0.01, split='va', FQE_save_dir=None, net_save_dir=None):
    X_list = data[..., -d:].reshape((-1, d))
    a_list = data[..., 1].astype(int)
    r_list = data[..., 2].astype(int)

    behavior_net = keras.models.load_model('{}/{}.behavior_net'.format(output_dir, split), compile=False)
    probs_b = behavior_probs = behavior_net.predict(X_list).reshape((-1, NSTEPS, nA))
    
    if FQE_save_dir is None:
        FQE_save_dir = '{}/NFQ-clipped-keras.{}FQE_models/'.format(output_dir, split)

    try:
        model_FQE = keras.models.load_model('{}/model={}.hidden_net'.format(FQE_save_dir, model_k), custom_objects={'select_output': select_output}, compile=False)
    except:
        return np.nan
    Q_pred = model_FQE.predict(X_list).reshape((-1, NSTEPS, nA))
    Q_pred[(a_list == -1) & (r_list == 0)] = 0 # zero out predictions after termination
    V_list = Q_pred.max(axis=-1)
    Q_pred = Q_pred.reshape((-1, nA))
    Q_list = Q_pred[range(len(Q_pred)), a_list.reshape((-1))].reshape((-1, NSTEPS))
    Q_pred = Q_pred.reshape((-1, NSTEPS, nA))

    if net_save_dir is None:
        net_save_dir = '{}/NFQ-clipped-keras.models/'.format(output_dir)
    
    hidden_net = keras.models.load_model('{}/iter={}.hidden_net'.format(net_save_dir, model_k), compile=False)
    a_pred = hidden_net.predict(X_list).reshape((-1, NSTEPS, nA)).argmax(axis=-1)

    return wdr_keras(data, V_list, Q_list, behavior_probs, a_pred, γ, epsilon)

def wdr_keras(data, V_list, Q_list, probs_b, a_pred, γ, epsilon):
    """Weighted Importance Sampling for Off Policy Evaluation
        - data: tensor of shape (N, T, 5) with the last last dimension being [t, s, a, r, s']
        - π_b:  behavior policy
        - π_e:  evaluation policy (aka target policy)
        - γ:    discount factor
    """
    t_list = data[..., 0].astype(int)
    a_list = data[..., 1].astype(int)
    r_list = data[..., 2].astype(int)
    
    # Per-transition importance ratios
    p_b = probs_b.reshape((-1, nA))[range(len(probs_b)*NSTEPS), a_list.reshape((-1))].reshape((-1, NSTEPS))
    p_e = (a_pred == a_list) * (1 - epsilon) + (a_pred != a_list) * (epsilon / (nA - 1)) # Get a soft version of the evaluation policy for WIS

    # Deal with variable length sequences by setting ratio to 1, value estimates to 0
    terminated_idx = (a_list == -1)
    p_b[terminated_idx] = 1
    p_e[terminated_idx] = 1
    
    if not np.all(p_b > 0):
        import pdb
        pdb.set_trace()
    assert np.all(p_b > 0), "Some actions had zero prob under p_b, WIS fails"

    # Per-trajectory cumulative importance ratios rho_{1:t} at each timestep
    rho_cum = (p_e / p_b).cumprod(axis=1)

    # Average cumulative importance ratio at every horizon t
    weights = rho_cum.mean(axis=0)
    
    # Weighted importance sampling ratios at each timestep
    w = rho_cum / weights
    w_1 = np.hstack([np.ones((w.shape[0],1)), w[..., :-1]]) # offset one timestep
    
    # Apply WDR estimator
    wdr_terms = np.power(γ, t_list) * (w_1 * V_list + w * r_list - w * Q_list)
    wdr_list = wdr_terms.sum(axis=1)
    wdr_value = wdr_list.mean()
    
    return wdr_value

def OPE_AM_keras(policy_value_net_k, X_init, γ, output_dir, save_dir=None, rollout=MAX_ROLLOUT, split='va'):
    reward_net = keras.models.load_model('{}/{}.dynamics.reward_net'.format(output_dir, split), compile=False, custom_objects={'select_output_d': select_output_d})
    delta_net = keras.models.load_model('{}/{}.dynamics.delta_net'.format(output_dir, split), compile=False, custom_objects={'select_output_d': select_output_d})
    
    if save_dir is None:
        save_dir = '{}/NFQ-clipped-keras.models/'.format(output_dir)
    
    policy_value_net = keras.models.load_model('{}/iter={}.hidden_net'.format(save_dir, policy_value_net_k), compile=False)
    
    X_rollouts, R_rollouts = [], []
    X_t = X_init
    X_rollouts.append(X_t)

    for t in range(rollout):
        A_t = policy_value_net.predict(X_t).argmax(axis=1)
        deltaX_t = delta_net.predict([X_t, A_t])
        newX_t = np.clip(X_t + deltaX_t, 0, 1)
        R_t = np.clip(reward_net.predict(newX_t), -1, 1)
        X_rollouts.append(newX_t)
        R_rollouts.append(R_t)

    R_all = np.concatenate(R_rollouts, axis=1)

    G_est1 = (
        np.cumprod(np.concatenate([np.ones((len(R_all), 1)), 1-np.abs(R_all)], axis=1)[:, :-1], axis=1) 
        * np.power(γ, range(rollout)) 
        * R_all
    ).sum(axis=1)

    G_est2 = (np.power(γ, range(rollout)) * R_all).sum(axis=1)

    return G_est1.mean(), G_est2.mean()
