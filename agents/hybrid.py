import numpy as np
from utils import softmax

"""
Hybrid agent similar to the one used in Daw et al. (2011)
Differences:
- Eligibility traces are not used
- Softmax perseveration parameter p is not used
- Therefore has 5 free parameters instead of 7
"""


class HybridAgent:
    def __init__(self, action_space, state_space, alpha_1=0.1, alpha_2=0.1, beta_1=0.9, beta_2=0.9, w=0):
        """
        Initialize hybrid agent
        :param action_space: Action space of the environment
        :param state_space: State space of the environment
        :param alpha_1: Learning rate for first step
        :param alpha_2: Learning rate for second step
        :param beta_1: Temperature parameter for softmax policy for first step
        :param beta_2: Temperature parameter for softmax policy for second step
        :param w: Weight of model-based values (0 for model-free, 1 for model-based)
        """
        self.action_space = action_space
        self.state_space = state_space
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.w = w

        self.q_td = np.zeros((len(state_space), len(action_space)))  # Q-table for model-free agent
        self.q_mb = np.zeros((len(state_space), len(action_space)))  # Q-table for model-based agent
        self.q_net = np.zeros((len(state_space), len(action_space)))  # Q-table for model-based agent
        self.q_table = self.q_net

        # Initialize transition model as a 3D numpy array
        # Dimensions: [current_state, action, next_state]
        # For simplicity, initializing all transitions as equally likely
        self.transition_model = np.zeros((len(state_space), len(action_space), len(state_space)))
        self.transition_counts = np.zeros((len(state_space), len(action_space), len(state_space)))

    def calculate_rpe(self, state, action, reward, next_state, next_action, terminal):
        """
        Calculate the reward prediction error
        :param state: current state
        :param action: current action
        :param reward: reward received in the current state
        :param next_state: next state
        :param next_action: next action
        :param terminal: whether this state is terminal
        :return:
        """
        if terminal:
            return reward - self.q_td[state, action]
        else:
            return reward + self.q_td[next_state, next_action] - self.q_td[state, action]

    def update_q_td(self, state, action, rpe):
        """
        Update the Q-table for model-free agent
        :param state: Current state
        :param action: Current action
        :param rpe: Reward prediction error
        :return:
        """
        alpha = self.alpha_1 if state == 0 else self.alpha_2
        self.q_td[state, action] += alpha * rpe

    def update_transition_model(self, current_state, action, next_state, terminal):
        if terminal:
            return

        # Increment the count for the observed transition
        self.transition_counts[current_state, action, next_state] += 1

        # Check if the transition is common or rare
        is_common_transition = self.transition_counts[current_state, action, next_state] == np.max(
            self.transition_counts[current_state, action])
        # Update the transition probabilities
        self.transition_model[current_state, action, next_state] = 0.7 if is_common_transition else 0.3
        other_states = np.array([state for state in self.state_space[1:] if state != next_state])
        for other_state in other_states:
            self.transition_model[current_state, action, other_state] = 0.3 if is_common_transition else 0.7

    def update_q_mb(self, state, action):
        if state == 0:
            self.q_mb[state, action] = np.sum(
                [self.transition_model[state, action, i] * np.max(self.q_td[i, :]) for i in self.state_space[1:]])
        else:
            self.q_mb[state, action] = self.q_td[state, action]

    def update_q_net(self, state, action):
        self.q_net[state, action] = self.w * self.q_mb[state, action] + (1 - self.w) * self.q_td[state, action]

    def update_beliefs(self, current_state, action, reward, next_state, terminal):
        next_action = self.policy(next_state)
        rpe = self.calculate_rpe(current_state, action, reward, next_state, next_action, terminal)
        self.update_q_td(current_state, action, rpe)
        self.update_transition_model(current_state, action, next_state, terminal)
        self.update_q_mb(current_state, action)
        self.update_q_net(current_state, action)

    def policy(self, state, method=None):
        beta = self.beta_1 if state == 0 else self.beta_2
        # TODO - Implement the softmax policy in the paper with parameter "p"
        action_probabilities = softmax(self.q_net[state, :], beta)
        return np.random.choice(self.action_space, p=action_probabilities)
