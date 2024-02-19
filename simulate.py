from datetime import datetime

import pandas as pd
import numpy as np
import os

from agents.model_based import AgentModelBased
from agents.model_free import AgentModelFree
from agents.random_agent import RandomAgent
from environment import TwoStepEnv
from utils import random_walk_gaussian


def simulate(agent_type='random', trials=200, **kwargs):
    # simulate the task
    action_space = TwoStepEnv.action_space
    state_space = TwoStepEnv.state_space

    if agent_type == 'model_based':
        agent = AgentModelBased(action_space, state_space, **kwargs)
    elif agent_type == 'model_free':
        agent = AgentModelFree(action_space, state_space, **kwargs)
    else:
        agent = RandomAgent(action_space, state_space, **kwargs)
    env = TwoStepEnv()
    task_data = simulate_two_step_task(env, agent, trials=trials)

    # (state, action) -> reward
    print("qtable:\n", agent.q_table)

    # (state, action, new state) -> transition probability
    if hasattr(agent, "transition_model"):
        # only the relevant transition probabilities should be non-zero, all others should be zero
        print("transition model for relevant states-action:\n",
              agent.transition_model[0])
        # print("transition model for other states-actions:\n",
        #       agent.transition_model[1:])

    # convert the data to a dataframe
    task_df = pd.DataFrame.from_dict(task_data, orient='index')
    task_df['trial_index'] = task_df.index

    # save the data to a csv file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join("data", "simulated", agent_type, timestamp)
    # Create folder if it does not exist
    os.makedirs(file_path, exist_ok=True)
    filename = os.path.join(file_path, "simulated_data.csv")
    task_df.to_csv(filename, index=False)

    return task_df

def simulate_two_step_task(env: TwoStepEnv, agent=None, trials=200,
                           policy_method="epsilon-greedy"):
    env.reset()
    task_data = {}

    sd_for_random_walk = 0.025
    time_step = 0
    while time_step < trials:
        # first stage choice
        terminal = False
        while not terminal:
            current_state = env.state
            if agent:
                action = agent.policy(env.state, method=policy_method)
            else:  # if no agent is given -> random action
                action = np.random.choice(env.action_space)

            next_state, reward, terminal, info = env.step(action)

            if agent:
                agent.update_beliefs(current_state, action, reward, next_state,
                                     terminal)

        task_data[time_step] = info
        env.reset()
        new_reward_prob_matrix = random_walk_gaussian(env.reward_prob_matrix,
                                                      sd_for_random_walk)
        env.set_reward_probabilities(new_reward_prob_matrix)
        time_step += 1

    return task_data
