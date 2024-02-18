from datetime import datetime

import pandas as pd
import os

from agents.model_based import AgentModelBased
from agents.model_free import AgentModelFree
from agents.random_agent import RandomAgent
from environment import TwoStepEnv
from utils import simulate_two_step_task


def simulate(agent_type='random'):
    # simulate the task
    if agent_type == 'model_based':
        agent = AgentModelBased(action_space=TwoStepEnv.action_space,
                                state_space=TwoStepEnv.state_space)
    elif agent_type == 'model_free':
        agent = AgentModelFree(action_space=TwoStepEnv.action_space,
                               state_space=TwoStepEnv.state_space)
    else:
        agent = RandomAgent(action_space=TwoStepEnv.action_space,
                            state_space=TwoStepEnv.state_space)
    env = TwoStepEnv()
    task_data = simulate_two_step_task(env, agent, trials=200)

    # (state, action) -> reward
    print("qtable:\n", agent.q_table)

    # (state, action, new state) -> transition probability
    if hasattr(agent, "transition_model"):
        # only the relevant transition probabilities should be non-zero, all others should be zero
        print("transition model for relevant states-action:\n",
              agent.transition_model[0])
        print("transition model for other states-actions:\n",
              agent.transition_model[1:])

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
