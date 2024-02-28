import numpy as np


class AgentModelFree:
    def __init__(self, action_space, state_space, alpha=0.1, beta=1.0, epsilon=0.2, gamma=0.9):
        # the state space can be infered but here it is given for simplicity
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.beta = beta  # Temperature parameter for softmax policy
        self.epsilon = epsilon  # Epsilon for epsilon-greedy policy
        self.q_table = np.zeros((len(self.state_space), len(self.action_space)))

    def softmax(self, arr, beta):
        e_x = np.exp(
            beta * (arr - np.max(arr)))  # subtract max value to prevent overflow
        return e_x / e_x.sum(
            axis=0)  # axis=0 for column-wise operation if arr is 2D, otherwise it's not needed

    def policy(self, state, beta=None, epsilon=None, method="softmax"):
        q_values = self.q_table[state, :]
        beta = self.beta if beta is None else beta
        epsilon = self.epsilon if epsilon is None else epsilon
        # calculate the probability of each action in the state with softmax
        if method == "softmax":
            action_probabilities = self.softmax(q_values, beta)
            action = np.random.choice(self.action_space, p=action_probabilities)

        # with epsilon gready policy
        else:
            if np.random.uniform() < epsilon:
                action = np.random.choice(self.action_space)
            else:
                action = np.argmax(q_values)

        return action

    def update_q_table_sarsa(self, state, action, reward, next_state, terminal):
        if state not in self.state_space or next_state not in self.state_space:
            raise ValueError(
                f"state:{state} is an invalid state, state space: {self.state_space}")
        if action not in self.action_space:
            raise ValueError(
                f"The action: {action} is not valid, action space: {self.action_space}")
        
        self.q_table[state, action] += self.alpha * self.reward_prediction_error(state,
                                                                                 action,
                                                                                 reward,
                                                                                 next_state,
                                                                                 terminal)

        return self.q_table

    def reward_prediction_error(self, state, action, reward, next_state, terminal):
        if terminal:
            return reward - self.q_table[state, action]

        next_action = self.policy(next_state)
        # # best action in the next state
        # next_action = np.argmax(self.q_table[next_state, :])
        return reward + self.gamma * self.q_table[next_state, next_action] - \
            self.q_table[state, action]
    
    def update_beliefs(self, state, action, reward, next_state, terminal):
        self.update_q_table_sarsa(state, action, reward, next_state, terminal)

    def get_action_probabilities(self, state):
        q_values = self.q_table[state, :]
        action_probabilities = self.softmax(q_values, self.beta)
        return action_probabilities

    def reset(self):
        self.q_table = np.zeros((len(self.state_space), len(self.action_space)))
        # return self.q_table