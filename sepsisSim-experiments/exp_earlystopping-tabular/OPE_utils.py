import numpy as np
import pandas as pd
import numpy_indexed as npi
import joblib
from tqdm import tqdm
import itertools
import copy

NSTEPS = 20       # max episode length in historical data
G_min = -1        # the minimum possible return
nS, nA = 1442, 8

##################
## Preparations ##
##################

def format_data_tensor(df_data, id_col='pt_id'):
    """
    Converts data from a dataframe to a tensor
    - df_data: pd.DataFrame with columns [id_col, Time, State, Action, Reward, NextState]
        - id_col specifies the index column to group episodes
    - data_tensor: integer tensor of shape (N, NSTEPS, 5) with the last last dimension being [t, s, a, r, s']
    """
    data_dict = dict(list(df_data.groupby(id_col)))
    N = len(data_dict)
    data_tensor = np.zeros((N, NSTEPS, 5), dtype=int)
    data_tensor[:, :, 2] = -1 # initialize all actions to -1
    data_tensor[:, :, 1] = -1 # initialize all states to -1
    data_tensor[:, :, 4] = -1 # initialize all next states to -1

    for i, (pt_id, df_values) in tqdm(enumerate(data_dict.items())):
        values = df_values.set_index(id_col).values
        data_tensor[i, :len(values), :] = values
    return data_tensor

def compute_empirical_MDP(df_data, id_col='pt_id'):
    """
    Calculate parameters of the empirical MDP (P, R, isd) 
    using Maximum Likelihood Estimation (MLE)
    """
    # Compute empirical MDP from data
    MDP_parameters = joblib.load('../data/MDP_parameters.joblib')
    P_true = MDP_parameters['transition_matrix_absorbing'] # (A, S, S_next)
    R_true = MDP_parameters['reward_matrix_absorbing_SA'] # (S, A)
    isd_true = joblib.load('../data/modified_prior_initial_state_absorbing.joblib')
    isd_true = (isd_true > 0)
    isd_true = isd_true / isd_true.sum()

    # Transition and Reward matrices
    transition_counts = df_data.groupby(['State', 'Action', 'NextState']).count()[['Reward']].rename(columns={'Reward': 'count'}).reset_index()
    P = P_true.copy()                 # retain for absorbing states
    P[:, (isd_true > 0), :] = 0       # estimate values for non-absorbing states
    R = R_true.copy()
    for i, row in transition_counts.iterrows():
        s, a, s_next = row['State'], row['Action'], row['NextState']
        count = row['count']
        if row['Action'] == -1:
            P[:, s, s_next] = count
        else:
            P[a, s, s_next] = count

    # assume unobserved (s,a) leads to death
    unobserved_sa_pairs = (P.sum(axis=-1) == 0)
    P[unobserved_sa_pairs, -2] = 1
    R[unobserved_sa_pairs.T] = -1

    # normalize transition probabilities
    P = P / P.sum(axis=-1, keepdims=True)

    # reset terminal absorbing states
    P[:, -2:, -2:] = P_true[:, -2:, -2:]

    # Initial state distribution
    initial_states = df_data.groupby(id_col)['State'].first()
    initial_states_count = initial_states.value_counts()
    isd = np.zeros_like(isd_true)
    for s in range(nS):
        if s in initial_states_count.index:
            isd[s] = initial_states_count[s]

    isd = isd / isd.sum()
    
    return P, R, isd

def compute_behavior_policy(df_data):
    """
    Calculate probabilities of the behavior policy π_b
    using Maximum Likelihood Estimation (MLE)
    """
    # Compute empirical behavior policy from data
    π_b = np.zeros((nS, nA))
    sa_counts = df_data.groupby(['State', 'Action']).count()[['Reward']].rename(columns={'Reward': 'count'}).reset_index()

    for i, row in sa_counts.iterrows():
        s, a = row['State'], row['Action']
        count = row['count']
        if row['Action'] == -1:
            π_b[s, :] = count
        else:
            π_b[s, a] = count

    # assume uniform action probabilities in unobserved states
    unobserved_states = (π_b.sum(axis=-1) == 0)
    π_b[unobserved_states, :] = 1

    # normalize action probabilities
    π_b = π_b / π_b.sum(axis=-1, keepdims=True)

    return π_b

def run_tabular_FQI(df_data, gamma, n_epochs, use_tqdm=False):
    """
    Tabular Fitted-Q Iteration
    """
    S, A, R, S_next = df_data['State'].values, df_data['Action'].values, df_data['Reward'].values, df_data['NextState'].values
    N = len(S)
    
    Qs = []
    Q = np.zeros((nS, nA))

    # unobserved states lead to minimum reward
    observed_s = set(S) | set(S_next) - {-1}
    unobserved_s = sorted(set(range(nS)) - observed_s)
    Q[unobserved_s, :] = G_min

    # unobserved action in observed states lead to minimum reward
    terminal_action = (A == -1)
    terminal_states = S[terminal_action]
    observed_sa = set(zip(S, A))
    unobserved_sa = np.array(sorted(
        set(itertools.product(observed_s, range(nA))) 
        - observed_sa - set(itertools.product(terminal_states, range(nA)))
    ))
    Q[unobserved_sa[:, 0], unobserved_sa[:, 1]] = G_min

    # reset absorbing states
    Q[-2:, :] = 0

    Qs.append(copy.deepcopy(Q))
    for k in tqdm(range(n_epochs), disable=not(use_tqdm)):
        y = R + gamma * Q[S_next, :].max(axis=1)
        
        # Update value as the sample average
        sa_list, value_list = npi.group_by(np.array([S, A]).T).mean(y)
        Q[sa_list[:, 0], sa_list[:, 1]] = value_list

        # Handle terminal states with action -1, applies to all actions
        terminal_action = (sa_list[:, 1] == -1)
        terminal_states = sa_list[terminal_action, :][:, 0]
        Q[terminal_states, :] = value_list[terminal_action][:,np.newaxis]

        # Save
        Qs.append(copy.deepcopy(Q))

    return Qs


#########################
## Evaluating a policy ##
#########################

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

def ground_truth_performance(π, γ):
    """
    Calculate the test-time performance using true MDP parameters
    - π: policy to be evaluated, shape (S,A)
    """
    # Ground truth MDP model
    MDP_parameters = joblib.load('../data/MDP_parameters.joblib')
    P = MDP_parameters['transition_matrix_absorbing'] # (A, S, S_next)
    R = MDP_parameters['reward_matrix_absorbing_SA'] # (S, A)

    # unif rand isd, mixture of diabetic state
    PROB_DIAB = 0.2
    isd = joblib.load('../data/modified_prior_initial_state_absorbing.joblib')
    isd = (isd > 0).astype(float)
    isd[:720] = isd[:720] / isd[:720].sum() * (1-PROB_DIAB)
    isd[720:] = isd[720:] / isd[720:].sum() * (PROB_DIAB)

    V_pi = policy_eval_analytic(P.transpose((1,0,2)), R, π, γ)
    return V_pi @ isd

def OPE_approx_model(π, P_approx, R_approx, isd_approx, γ):
    """
    - π: policy to be evaluated, shape (S,A)
    - P_approx, R_approx, isd_approx: approximate MDP model parameters estimated from data
    """
    V_pi = policy_eval_analytic(P_approx.transpose((1,0,2)), R_approx, π, γ)
    return V_pi @ isd_approx

def OPE_WIS(data, π_b, π_e, γ, epsilon=0.01, bootstrap=None):
    """
    - π_b, π_e: behavior/evaluation policy, shape (S,A)
    """
    # Get a soft version of the evaluation policy for WIS
    π_e_soft = np.copy(π_e).astype(float)
    π_e_soft[π_e_soft == 1] = (1 - epsilon)
    π_e_soft[π_e_soft == 0] = epsilon / (nA - 1)
    
    # Apply WIS
    return _wis(data, π_b, π_e_soft, γ, bootstrap)

def _wis(data, π_b, π_e, γ, bootstrap):
    """
    Weighted Importance Sampling for Off-Policy Evaluation
        - data: tensor of shape (N, T, 5) with the last last dimension being [t, s, a, r, s']
        - π_b:  behavior policy
        - π_e:  evaluation policy (aka target policy)
        - γ:    discount factor
    """
    t_list = data[..., 0].astype(int)
    s_list = data[..., 1].astype(int)
    a_list = data[..., 2].astype(int)
    r_list = data[..., 3].astype(int)
    
    # Per-trajectory returns (discounted cumulative rewards)
    G = (r_list * np.power(γ, t_list)).sum(axis=-1)
    
    # Per-transition importance ratios
    p_b = π_b[s_list, a_list]
    p_e = π_e[s_list, a_list]

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

    if bootstrap is None:
        # directly calculate weighted average over trajectories
        wis_value = np.average(G, weights=rho)
        return wis_value, (rho > 0).sum(), rho.sum()
    else:
        # Get indices for bootstrap, because we need to sample from rho and G
        idx = np.random.default_rng(seed=0).choice(np.arange(len(G)), size=(bootstrap, len(G)), replace=True)
        wis_boot = (rho[idx] / rho[idx].mean(axis=1, keepdims=True)) * G[idx] 
#         wis_boot = [np.average(G[idx[i]], weights=rho[idx[i]]) for i in range(bootstrap)]
        
        # Return WIS, one per row
        wis_value = wis_boot.mean(axis=1)
        return wis_value, (rho[idx] > 0).sum(axis=1), rho[idx].sum(axis=1)

def run_tabular_FQE(df_data, π, gamma, n_epochs, use_tqdm=False):
    """
    Tabular Fitted-Q Evaluation
        - π (S,A): policy to be evaluated
    """
    S, A, R, S_next = df_data['State'].values, df_data['Action'].values, df_data['Reward'].values, df_data['NextState'].values
    N = len(S)

    Q = np.zeros((nS, nA))

    # unobserved states lead to minimum reward
    observed_s = set(S) | set(S_next) - {-1}
    unobserved_s = sorted(set(range(nS)) - observed_s)
    Q[unobserved_s, :] = G_min

    # unobserved action in observed states lead to minimum reward
    terminal_action = (A == -1)
    terminal_states = S[terminal_action]
    observed_sa = set(zip(S, A))
    unobserved_sa = np.array(sorted(
        set(itertools.product(observed_s, range(nA))) 
        - observed_sa - set(itertools.product(terminal_states, range(nA)))
    ))
    Q[unobserved_sa[:, 0], unobserved_sa[:, 1]] = G_min

    # reset absorbing states
    Q[-2:, :] = 0

    for k in tqdm(range(n_epochs), disable=not(use_tqdm)):
        y = R + gamma * (π[S_next, :] * Q[S_next, :]).sum(axis=1)

        # Update value as the sample average
        sa_list, value_list = npi.group_by(np.array([S, A]).T).mean(y)
        Q[sa_list[:, 0], sa_list[:, 1]] = value_list

        # Handle terminal states with action -1, applies to all actions
        terminal_action = (sa_list[:, 1] == -1)
        terminal_states = sa_list[terminal_action, :][:, 0]
        Q[terminal_states, :] = value_list[terminal_action][:,np.newaxis]

    return Q

def OPE_FQE(df_data, isd, π, n_epochs=50, use_tqdm=False):
    Q = run_tabular_FQE(df_data, π, n_epochs, use_tqdm)
    return isd @ (Q * π).sum(axis=1)

def V2Q(V, P, R, gamma):
    nS, nA = R.shape
    Q = np.zeros((nS, nA))
    for a in range(nA):
        Q[:,a] = sum(P[:,a,s_] * (R[:,a] + gamma * V[s_]) for s_ in range(nS))
    return Q

def OPE_WDR(data, V, Q, π_b, π_e, γ, epsilon=0.01):
    # Get a soft version of the evaluation policy for WIS
    π_e_soft = np.copy(π_e).astype(float)
    π_e_soft[π_e_soft == 1] = (1 - epsilon)
    π_e_soft[π_e_soft == 0] = epsilon / (nA - 1)
    
    # Apply WDR
    return _wdr(data, V, Q, π_b, π_e_soft, γ)

def _wdr(data, V, Q, π_b, π_e, γ):
    """
    Weighted Doubly Robust for Off Policy Evaluation
    * Reference: http://proceedings.mlr.press/v48/thomasa16.html
        - data: tensor of shape (N, T, 5) with the last last dimension being [t, s, a, r, s']
        - V, Q: estimated value functions
        - π_b:  behavior policy
        - π_e:  evaluation policy (aka target policy)
        - γ:    discount factor
    """
    t_list = data[..., 0].astype(int)
    s_list = data[..., 1].astype(int)
    a_list = data[..., 2].astype(int)
    r_list = data[..., 3].astype(int)

    # Per-transition importance ratios
    p_b = π_b[s_list, a_list]
    p_e = π_e[s_list, a_list]

    # Deal with variable length sequences by setting ratio to 1
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
    wdr_terms = np.power(γ, t_list) * (w_1 * V[s_list] + w * r_list - w * Q[s_list, a_list])
    wdr_list = wdr_terms.sum(axis=1)
    wdr_value = wdr_list.mean()
    
    return wdr_value
