# simulate data
# (for now from random agent, as test the environment and task implementation)
import numpy as np
from environment import TwoStepEnv


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


def random_walk_gaussian(prob, sd, min_prob=0, max_prob=1):
    new_prob = prob + np.random.normal(scale=sd, size=np.shape(prob))
    new_prob = np.clip(new_prob, min_prob, max_prob)
    return new_prob
