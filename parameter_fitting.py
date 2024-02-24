from simulate import simulate
from agents.random_agent import RandomAgent
from agents.model_free import AgentModelFree
from agents.model_based import AgentModelBased
from environment import TwoStepEnv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

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

def average_accumulated_reward(data):
    return data['reward'].mean()

def fit_to_average_accumulated_reward(parammeter_space: dict, agent_type='randome', seed=0, verbose=False):
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
            accumulated_reward_results[idx_1, idx_2] = average_accumulated_reward(data)
    return accumulated_reward_results


def log_likelihood(agent, data, consider_both_stages=True, verbose=False):
    LogLikelihood_sum = 0
    for idx ,trail_data in data.iterrows():
        chosen_action_1 = trail_data['stepOneChoice']
        chosen_action_2 = trail_data['stepTwoChoice']
        stage_1_state = 0
        stage_2_state = trail_data['state_transition_to']
        recieved_reward = int(trail_data['reward'])

        action_probs_stage_1 = get_action_probs(agent, stage_1_state) # return a list of action probabilities
        chosen_action_1_prob = action_probs_stage_1[chosen_action_1]
        # TODO updtae belief atter each action
        action_probs_stage_2 = get_action_probs(agent, stage_2_state)
        chosen_action_2_prob = action_probs_stage_2[chosen_action_2]

        # calculate the log likelihood
        # based on only the first stage stage
        LogLikelihood_sum += np.log(chosen_action_1_prob)
        
        if consider_both_stages:
            # based on both stages
            # assuming the actions are independent following the Markov property
            # P(a1, a2) = P(a1) * P(a2)
            # -> log(P(a1, a2)) = log(P(a1)) + log(P(a2))
            LogLikelihood_sum += np.log(chosen_action_2_prob)
        
        # let the model choose the actions of the human and update its beliefs
        stage_1 = (stage_1_state, chosen_action_1)
        stage_2 = (stage_2_state, chosen_action_2)

        # update the model's beliefs
        apply_choices(agent, stage_1, stage_2, recieved_reward)

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

def LL_for_params_search_space(parammeter_space: dict, human_data, agent_type='randome', consider_both_stages=True, verbose=False):
    print(f'agent_type: {agent_type}, consider_both_stages: {consider_both_stages}')
    param_names = [name for name in list(parammeter_space.keys())]
    param_values = [values for values in list(parammeter_space.values())]
    n1 = param_names[0]
    n2 = param_names[1]
    LL_results = np.zeros((len(param_values[0]), len(param_values[1])))
    for idx_1, p1 in enumerate(param_values[0]):
        for idx_2, p2 in enumerate(param_values[1]):
            param_dict = {n1: p1,
                        n2: p2}
            if agent_type == 'model_free':
                agent = AgentModelFree(TwoStepEnv.action_space, TwoStepEnv.state_space, **param_dict)
            elif agent_type == 'model_based':
                agent = AgentModelBased(TwoStepEnv.action_space, TwoStepEnv.state_space, **param_dict)
            else:
                agent = RandomAgent(TwoStepEnv.action_space, TwoStepEnv.state_space)

            LL_results[idx_1, idx_2] = log_likelihood(agent, human_data, consider_both_stages, verbose)
    
    return LL_results

# plot the log likelihoods
def plot_fit_results(LL_results, alpha_space, beta_space, full=False):
    if full:
        plt.figure(figsize=(10, 6))
        for idx, beta in enumerate(beta_space):
            plt.plot(alpha_space, LL_results[idx,:], label=f'alpha = {beta}')
        plt.xlabel('beta')
        plt.ylabel('Log Likelihood')
        plt.title('Log Likelihood of human data given model')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 6))
        for idx, alpha in enumerate(alpha_space):
            plt.plot(beta_space, LL_results[:,idx], label=f'beta = {alpha}')
        plt.xlabel('alpha')
        plt.ylabel('Log Likelihood')
        plt.title('Log Likelihood of human data given model')
        plt.legend()
        plt.show()

    # plot color map of the log likelihoods
    # extend the results to a 2D space with a dummy dimension
    # LL_results = np.expand_dims(LL_results, axis=0)
    x_ticks = np.round(beta_space, 3)
    y_ticks = np.round(alpha_space, 3)
    visible_ticks_x = [x_ticks[0], x_ticks[len(x_ticks) // 2], x_ticks[-1]]
    visible_ticks_y = [y_ticks[0], y_ticks[len(y_ticks) // 2], y_ticks[-1]]
    plt.figure(figsize=(10, 6))
    plt.imshow(LL_results, origin='lower', cmap='inferno')
    # plt.imshow(LL_results, origin='lower', cmap='viridis')
    plt.xlabel('beta')
    plt.xticks([0, len(x_ticks) // 2, len(x_ticks) - 1], visible_ticks_x)
    plt.ylabel('alpha')
    plt.yticks([0, len(y_ticks) // 2, len(y_ticks) - 1], visible_ticks_y)
    plt.colorbar(label='Log Likelihood')
    plt.title('Log Likelihoods')
    # Annotate the maximum log likelihood on the plot
    max_ll_idx = np.unravel_index(np.argmax(LL_results, axis=None), LL_results.shape)
    max_alpha = alpha_space[max_ll_idx[0]]
    max_beta = beta_space[max_ll_idx[1]]
    max_alpha = np.round(max_alpha, 2)
    max_beta = np.round(max_beta, 2)
    plt.annotate(f'Max LL\nalpha={max_alpha}, beta={max_beta}',
             xy=(max_ll_idx[1], max_ll_idx[0]),
             xytext=(max_ll_idx[1]+1, max_ll_idx[0]+1),
             arrowprops=dict(facecolor='white', shrink=0.005), color='darkgreen')
    plt.show()

