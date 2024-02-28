import numpy as np

class RandomAgent:
    def __init__(self, action_space, state_space):
        # the state space can be infered but here it is given for simplicity
        self.action_space = action_space
        self.state_space = state_space

    def policy(self, state, method=None):
        return np.random.choice(self.action_space)

    def update_beliefs(self, state, action, reward, next_state, terminal):
        return

    def reset(self):
        return

    def get_action_probabilities(self, state):
        return np.ones(len(self.action_space)) / len(self.action_space)
