import numpy as np


class RandomAgent:
    def __init__(self, action_space, state_space, alpha=0.1, gamma=0.9):
        # the state space can be infered but here it is given for simplicity
        self.action_space = action_space
        self.state_space = state_space
        self.q_table = np.zeros((len(self.state_space), len(self.action_space)))
        self.alpha = alpha
        self.gamma = gamma

    def policy(self, state, method=None):
        return np.random.choice(self.action_space)

    def update_q_table_sarsa(self, state, action, reward, next_state, terminal):
        if state not in self.state_space or next_state not in self.state_space:
            raise ValueError(
                f"state:{state} is an invalid state, state space: {self.state_space}")
        if action not in self.action_space:
            raise ValueError(
                f"The action: {action} is not valid, action space: {self.action_space}")

        if terminal:
            self.q_table[state, action] += self.alpha * (
                    reward - self.q_table[state, action])
        else:
            next_action = self.policy(next_state)
            self.q_table[state, action] += self.alpha * (
                    reward + self.gamma * self.q_table[next_state, next_action] -
                    self.q_table[state, action])
        return self.q_table

    def update_beliefs(self, state, action, reward, next_state, terminal):
        self.update_q_table_sarsa(state, action, reward, next_state, terminal)

    def reset(self):
        pass

    def get_action_probabilities(self, state):
        return np.ones(len(self.action_space)) / len(self.action_space)
