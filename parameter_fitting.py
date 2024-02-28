from simulate import simulate
from agents.random_agent import RandomAgent
from agents.model_free import AgentModelFree
from agents.model_based import AgentModelBased
from agents.hybrid import HybridAgent
from environment import TwoStepEnv
from sklearn.model_selection import ParameterSampler
from itertools import product
from scipy.optimize import minimize
from scipy.stats import uniform
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
# from tqdm import tqdm
# import the tqdm library combpatibilitle with jupyter notebook
from tqdm.notebook import tqdm

# calculate the log likelihood of the human data given the model
def get_action_probs(agent=None, state=None):
    if agent is None:
      return np.random.uniform(size=2)

    return agent.get_action_probabilities(state)

def apply_choices(agent=None, stage_1=None, stage_2=None, recieved_reward=None):
    if agent is None:
        return

    state_1 = stage_1[0]
    action_1 = stage_1[1]
    state_2 = stage_2[0]
    action_2 = stage_2[1]
    # state, action, reward, next_state, terminal
    agent.update_beliefs(state_1, action_1, 0, state_2, False)
    agent.update_beliefs(state_2, action_2, recieved_reward, state_2, True)

def log_likelihood(agent, data, consider_both_stages=True, verbose=False, show_progress=False):
    LogLikelihood_sum = 0
    for idx, trial_data in tqdm(data.iterrows(), desc='log_likelihood:', total=len(data), disable=not show_progress, leave=False):
        chosen_action_1 = trial_data['stepOneChoice']
        chosen_action_2 = trial_data['stepTwoChoice']
        stage_1_state = 0
        stage_2_state = trial_data['state_transition_to']
        recieved_reward = int(trial_data['reward'])

        action_probs_stage_1 = get_action_probs(agent, stage_1_state)
        chosen_action_1_prob = action_probs_stage_1[chosen_action_1]
        agent.update_beliefs(stage_1_state, chosen_action_1, 0, stage_2_state, False)

        action_probs_stage_2 = get_action_probs(agent, stage_2_state)
        chosen_action_2_prob = action_probs_stage_2[chosen_action_2]
        agent.update_beliefs(stage_2_state, chosen_action_2, recieved_reward, stage_2_state, True)

        # calculate the log likelihood
        # based on only the first stage stage
        LogLikelihood_sum += np.log(chosen_action_1_prob)

        if consider_both_stages:
            # based on both stages
            # assuming the actions are independent following the Markov property
            # P(a1, a2) = P(a1) * P(a2)
            # -> log(P(a1, a2)) = log(P(a1)) + log(P(a2))
            LogLikelihood_sum += np.log(chosen_action_2_prob)

        # print everything for debugging
        if verbose:
            print(f'reward recieved: {recieved_reward}')
            print(f'chosen_action_1: {chosen_action_1}')
            print(f'chosen_action_2: {chosen_action_2}')
            print(f'stage_2_state: {stage_2_state}')
            print(f'action_probs_stage_1: {action_probs_stage_1}')
            print(f'action_probs_stage_2: {action_probs_stage_2}')
            print(f'q_table: {agent.q_table}')
            if hasattr(agent, 'transition_model'):
                print(f'transition_matrix: {agent.transition_model[0]}')
            print('#'*50)

    return LogLikelihood_sum

def fit_with_MCMC(parammeter_space: dict, data, agent_type, consider_both_stages=True, num_samples=1000, num_burn_in=1000,
                    verbose=False, show_progress=False):
    sampling_sd = 0.1
    param_names = list(parammeter_space.keys())
    param_bounds = [(np.min(parammeter_space[param]),np.max(parammeter_space[param])) for param in param_names]

    def log_likelihood_function(params):
        # create the parameter dictionary
        params = dict(zip(param_names, params))
        agent = create_agent(agent_type, params)
        return log_likelihood(agent, data, consider_both_stages)
    
    samples = np.zeros((num_samples, len(param_names)))
    initial_guess = [np.mean(bounds) for bounds in param_bounds] # initial guess varaiable to keep the code readable
    current_params = np.array(initial_guess)
    log_likelihood_values = []
    for i in tqdm(range(num_samples + num_burn_in), desc='MCMC:', total=num_samples + num_burn_in, disable=not show_progress, leave=True):
        # sample new parameters
        new_params = np.random.normal(current_params, scale=sampling_sd, size=current_params.shape)
        # bound the parameters to the parameter space
        new_params = np.clip(new_params, [bound[0] for bound in param_bounds], [bound[1] for bound in param_bounds])
        # calculate the log likelihood for the new parameters
        log_likelihood_current = log_likelihood_function(current_params)
        log_likelihood_new = log_likelihood_function(new_params)
        # calculate the acceptance probability
        accept_prob = np.exp(log_likelihood_new - log_likelihood_current)
        # accept or reject the new parameters
        if log_likelihood_new > log_likelihood_current or np.random.rand() < accept_prob:
            current_params = new_params
            if i >= num_burn_in:
                log_likelihood_values.append(log_likelihood_new)
        elif i >= num_burn_in:
            log_likelihood_values.append(log_likelihood_current)
        # store the parameters after the burn in period
        if i >= num_burn_in:
            samples[i - num_burn_in] = current_params
    # Convert to DataFrame
    results_df = pd.DataFrame(samples, columns=param_names)
    results_df['log_likelihood'] = log_likelihood_values
    # get the best parameters and the best LL
    best_params, best_LL = get_best_params_and_ll(results_df)
    return best_params, best_LL, results_df


def fit_with_MCMC(parammeter_space, data, agent_type, consider_both_stages=True, num_samples=1000, num_burn_in=100, verbose=False, show_progress=False):
    sampling_sd = 0.1
    param_names = list(parammeter_space.keys())
    param_bounds = [(np.min(parammeter_space[param]),np.max(parammeter_space[param])) for param in param_names]
    initial_guess = [np.mean(bounds) for bounds in param_bounds]
    # Define the log likelihood function
    def log_likelihood_function(params):
        params_dict = dict(zip(param_names, params))
        agent = create_agent(agent_type, params_dict)
        return log_likelihood(agent, data, consider_both_stages)
    
    # Initialize the sampling process
    samples = np.zeros((num_samples, len(param_names)))
    current_params = np.array([np.mean(bounds) for bounds in param_bounds])
    log_likelihood_values = []
    
    for i in tqdm(range(num_samples + num_burn_in), desc='MCMC Sampling', disable=not show_progress):
        # Propose new parameters
        proposal_params = current_params + np.random.normal(0, sampling_sd, size=len(param_names))
        
        # Enforce parameter bounds
        proposal_params = np.clip(proposal_params, [b[0] for b in param_bounds], [b[1] for b in param_bounds])
        
        # Calculate log likelihoods
        log_likelihood_current = log_likelihood_function(current_params)
        log_likelihood_proposal = log_likelihood_function(proposal_params)
        
        # Acceptance probability
        accept_prob = np.exp(log_likelihood_proposal - log_likelihood_current)
        
        if log_likelihood_proposal > log_likelihood_current or np.random.rand() < accept_prob:
            current_params = proposal_params
            if i >= num_burn_in:
                log_likelihood_values.append(log_likelihood_proposal)
        elif i >= num_burn_in:
            # If not accepted, repeat the current parameters' log likelihood
            log_likelihood_values.append(log_likelihood_current)
        
        if i >= num_burn_in:
            samples[i - num_burn_in] = current_params
    
    # Convert samples and log likelihoods to a DataFrame
    results_df = pd.DataFrame(samples, columns=param_names)
    results_df['log_likelihood'] = log_likelihood_values
    
    # Assuming get_best_params_and_ll extracts the best parameters and the highest log likelihood
    best_params, best_LL = get_best_params_and_ll(results_df)
    
    return best_params, best_LL, results_df


def fit_with_minimize(parammeter_space: dict, data, agent_type, consider_both_stages=True, num_initializations=10,
                    verbose=False, show_progress=False):
    param_names = list(parammeter_space.keys())
    param_bounds = [(np.min(parammeter_space[param]),np.max(parammeter_space[param])) for param in param_names]
    # take the mean of the bounds as the initial guess
    def objective_function(params):
        # create the parameter dictionary
        params = dict(zip(param_names, params))
        agent = create_agent(agent_type, params)
        return -log_likelihood(agent, data, consider_both_stages)

    # run the optimization multiple from different starting points and take the best result 
    best_LL = np.inf
    best_params = None
    sampled_results = []
    for i in tqdm(range(num_initializations), desc='initializations:', total=num_initializations, disable=not show_progress, leave=True):
        if i == 0:
            initial_guess = [np.mean(bounds) for bounds in param_bounds]
        else:
            initial_guess = [np.random.uniform(low, high) for low, high in param_bounds]
        LL_result = minimize(objective_function, initial_guess, bounds=param_bounds)
        sampled_results.append({**dict(zip(param_names, LL_result.x)), 'log_likelihood': -LL_result.fun})
        if LL_result.fun < best_LL:
            best_LL = LL_result.fun
            best_params = dict(zip(param_names, LL_result.x))
        if verbose:
            print(f'run: {i}, LL: {-LL_result.fun}, params: {dict(zip(param_names, LL_result.x))}')
            print('#'*50)

    # Convert to DataFrame
    results_df = pd.DataFrame(sampled_results)
    return best_params, -best_LL, results_df

# LL_for_params_search_space
def fit_with_grid_search(parammeter_space: dict, data, agent_type, consider_both_stages=True,
                        verbose=False, show_progress=False):
    param_names = list(parammeter_space.keys())
    param_combinations = list(product(*parammeter_space.values()))
    LL_results = np.zeros(len(param_combinations))
    
    for idx, param_vals in tqdm(enumerate(param_combinations), desc='grid_search:', total=len(param_combinations), disable=not show_progress, leave=True):
        param_dict = dict(zip(param_names, param_vals))
        agent = create_agent(agent_type, param_dict)
        LL_results[idx] = log_likelihood(agent, data, consider_both_stages, verbose)
    
    # Reshape the results to fit the dimensions of the parameter space
    shape_dims = [len(values) for values in parammeter_space.values()]
    LL_results = LL_results.reshape(*shape_dims)
    # get the best parameters and the best LL
    best_params_idx = np.unravel_index(LL_results.argmax(), LL_results.shape)
    best_params = {param: parammeter_space[param][best_params_idx[i]] for i, param in enumerate(parammeter_space.keys())}
    best_LL = LL_results[best_params_idx]
    return best_params, best_LL, LL_results

def fit_with_random_search(param_space:dict, data, agent_type, num_iterations=100,consider_both_stages=True, seed=0,
                           verbose=False, show_progress=False):
    # convert the parameter space to a format that can be used by ParameterSampler
    # scipy.stats.uniform ->  (loc, loc + scale)
    param_distribution = {param: uniform(np.min(param_space[param]), np.max(param_space[param]) - np.min(param_space[param]))
                    for param in param_space.keys()}
    # Generate parameter samples
    param_list = list(ParameterSampler(param_distribution, n_iter=num_iterations, random_state=seed))

    # Initialize an empty list for the results
    sampled_results = []

    # Evaluate log likelihood for sampled parameter sets
    for params in tqdm(param_list, desc='random_search:', total=len(param_list), disable=not show_progress, leave=True):
        agent = create_agent(agent_type, params)
        log_likelihood_value = log_likelihood(agent, data, consider_both_stages)
        sampled_results.append({**params, 'log_likelihood': log_likelihood_value})

    # Convert to DataFrame
    results_df = pd.DataFrame(sampled_results)
    # get the best parameters and the best LL
    best_params, best_LL = get_best_params_and_ll(results_df)
    return best_params, best_LL, results_df

def get_best_params_and_ll(results_df):
    # Sort the dataframe and get first row
    sorted_df = results_df.sort_values(by='log_likelihood', ascending=False)
    best_parameters_row = sorted_df.iloc[0]

    # Extract the best parameters and log likelihood
    best_params = best_parameters_row.drop('log_likelihood').to_dict()
    best_log_likelihood = best_parameters_row['log_likelihood']

    return best_params, best_log_likelihood

def create_agent(agent_type, params):
    if agent_type == 'model_free':
        agent = AgentModelFree(TwoStepEnv.action_space, TwoStepEnv.state_space, **params)
    elif agent_type == 'model_based':
        agent = AgentModelBased(TwoStepEnv.action_space, TwoStepEnv.state_space, **params)
    elif agent_type == 'hybrid' or agent_type.startswith('hybrid'):
        agent = HybridAgent(TwoStepEnv.action_space, TwoStepEnv.state_space, **params)
    else:
        agent = RandomAgent(TwoStepEnv.action_space, TwoStepEnv.state_space, **params)
    return agent

def fit_to_average_cumulative_reward(parammeter_space: dict, agent_type='randome', seed=None, verbose=False):
    param_names = [name for name in list(parammeter_space.keys())]
    param_values = [values for values in list(parammeter_space.values())]
    n1 = param_names[0]
    n2 = param_names[1]
    accumulated_reward_results = np.zeros((len(param_values[0]), len(param_values[1])))
    for idx_1, p1 in enumerate(param_values[0]):
        for idx_2, p2 in enumerate(param_values[1]):
            param_dict = {n1: p1,
                        n2: p2}
            data, _ = simulate(agent_type=agent_type, seed=seed, params=param_dict)
            accumulated_reward_results[idx_1, idx_2] = data['reward'].mean()
    return accumulated_reward_results

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# plotting functions

# plot the log likelihoods
def plot_fit_results(LL_results:np.ndarray, parameter_space:dict, full=False, title='', save=False, filename='plots/fit_results.png'):
    num_params = len(parameter_space)
    if num_params == 1:
        plot_fit_results_1d(LL_results, parameter_space, full=full, title=title, save=save, filename=filename)

    elif num_params == 2:
        plot_fit_results_2d(LL_results, parameter_space, full=full, title=title, save=save, filename=filename)
    
    elif num_params > 2:
        plot_heatmap_slices(LL_results, parameter_space, full=full, title=title, save=save, filename=filename)

    else:
        print(f'parameter space should be of type dict, provided: {type(parameter_space)}')

def plot_fit_results_1d(LL_results:np.ndarray, parameter_space:dict, full=False, title='', save=False, filename='plots/fit_results.png'):
    param_name, param_space = list(parameter_space.items())[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title)
    ax.plot(param_space, LL_results)
    ax.set_xlabel(param_name)
    ax.set_ylabel('Log Likelihood')
    ax.set_title('Log Likelihood of human data given model')
    fig.tight_layout()
    plt.show()
    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename.replace('.png', f'_{timestamp}.png')
        fig.savefig(filename)
        print(f'Plot saved to {filename}')

def plot_fit_results_2d(LL_results, parameter_space:dict, full=False, title='', save=False, filename='plots/fit_results.png'):
    if len(parameter_space) != 2:
        raise ValueError(f'This function is only for 2D parameter spaces, provided {len(parameter_space)}D')
    
    seperated_param_space = list(parameter_space.items())
    param_1, param_1_space = seperated_param_space[0]
    param_2, param_2_space = seperated_param_space[1]
    
    if full:
        # plot the log likelihoods for each parameter value across the other parameter
        fig, (ax_1, ax_2) = plt.figure(nrows=1, ncols=2, figsize=(10, 6))
        for idx, val in enumerate(param_1_space):
            ax_1.plot(param_2_space, LL_results[:,idx], label=f'{param_1} = {val}')
        ax_1.set_xlabel(param_2)
        ax_1.set_ylabel('Log Likelihood')
        ax_1.set_title('Log Likelihood of human data given model')
        ax_1.legend()

        for idx, val in enumerate(param_2_space):
            ax_2.plot(param_1_space, LL_results[idx,:], label=f'{param_2} = {val}')
        ax_2.set_xlabel(param_1)
        ax_2.set_ylabel('Log Likelihood')
        ax_2.set_title('Log Likelihood of human data given model')
        ax_2.legend()

        fig.tight_layout()
        plt.show()
        if save:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            filename = filename.split('.')[0] + '_full.png'
            # add timestamp to filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filename.replace('.png', f'_{timestamp}.png')
            fig.savefig(filename)
            print(f'Plot saved to {filename}')

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(title)

    x_ticks = np.round(param_2_space, 3)
    y_ticks = np.round(param_1_space, 3)
    visible_ticks_x = [x_ticks[0], x_ticks[len(x_ticks) // 2], x_ticks[-1]]
    visible_ticks_y = [y_ticks[0], y_ticks[len(y_ticks) // 2], y_ticks[-1]]

    im = ax.imshow(LL_results, origin='lower', cmap='inferno')

    ax.set_xlabel(param_2)
    ax.set_xticks([0, len(x_ticks) // 2, len(x_ticks) - 1])
    ax.set_xticklabels(visible_ticks_x)
    ax.set_ylabel(param_1)
    ax.set_yticks([0, len(y_ticks) // 2, len(y_ticks) - 1])
    ax.set_yticklabels(visible_ticks_y)
    ax.set_title('Log Likelihoods')

    fig.colorbar(im, ax=ax, label='Log Likelihood')

    # Annotate the maximum log likelihood on the plot
    max_ll_idx = np.unravel_index(np.argmax(LL_results, axis=None), LL_results.shape)
    max_ll = np.round(LL_results[max_ll_idx], 2)
    max_param_1 = np.round(param_1_space[max_ll_idx[0]], 2)
    max_param_2 = np.round(param_2_space[max_ll_idx[1]], 2)

    ax.annotate(f'Max LL={max_ll}\n{param_1}={max_param_1}, {param_2}={max_param_2}',
            xy=(max_ll_idx[1], max_ll_idx[0]),
            xytext=(max_ll_idx[1]+1, max_ll_idx[0]+1),
            arrowprops=dict(facecolor='darkgreen', shrink=0.005), color='darkgreen', zorder=10, textcoords='offset points')
    
    fig.tight_layout()
    plt.show()
    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename.replace('.png', f'_{timestamp}.png')
        fig.savefig(filename)
        print(f'Plot saved to {filename}')

def plot_heatmap_slices(LL_results, parameter_space, full=False, title='', save=False, filename='plots/fit_results.png'):
    num_params = len(parameter_space)
    param_names = list(parameter_space.keys())
    param_spaces = [parameter_space[name] for name in param_names]
    
    # Find the indices of the maximum log likelihood
    max_ll_indices = np.unravel_index(np.argmax(LL_results), LL_results.shape)
    # plot the 2D slices for all pairs of parameters and fixed values for the other parameters at the maximum LL
    for i in range(num_params):
        for j in range(i + 1, num_params):
            # Extract the 2D slice for the i-th and j-th parameters
            fixed_params = list(max_ll_indices)
            fixed_params[i] = slice(None)  # All values for the i-th param
            fixed_params[j] = slice(None)  # All values for the j-th param
            slice_ij = LL_results[tuple(fixed_params)]

            # define the 2d parameter space and plot using plot_fit_results_2d
            param_space_ij = {param_names[i]: param_spaces[i], 
                              param_names[j]: param_spaces[j]}
            # pass the fixed parameters and their values as title
            plot_title = f'{title} {param_names[i]}={param_spaces[i][max_ll_indices[i]]}, {param_names[j]}={param_spaces[j][max_ll_indices[j]]}'
            if save:
                filename = filename.split('.')[0] + f'_{param_names[i]}_{param_spaces[i][max_ll_indices[i]]}_{param_names[j]}_{param_spaces[j][max_ll_indices[j]]}.png'
            plot_fit_results_2d(slice_ij, param_space_ij, full, plot_title, save, filename)