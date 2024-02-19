import numpy as np


class AgentModelBased:
    def __init__(self, action_space, state_space, alpha=0.1, gamma=0.9, beta=1.0,
                 epsilon=0.2):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.beta = beta  # Temperature parameter for softmax policy
        self.epsilon = 0.2  # Epsilon for epsilon-greedy policy
        self.q_table = np.zeros((len(state_space), len(action_space)))

        # Initialize transition model as a 3D numpy array
        # Dimensions: [current_state, action, next_state]
        # For simplicity, initializing all transitions as equally likely
        self.transition_model = np.zeros((len(state_space),
                                          len(action_space),
                                          len(state_space)))
        self.transition_counts = np.zeros((len(state_space),
                                           len(action_space),
                                           len(state_space)))

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

    def update_transition_model(self, current_state, action, next_state, terminal):
        # Simple counting method to update transition probabilities
        # TODO - Implement more sophisticated methods like Bayesian updating
        #      - at least insure no 0 probabilities for stage 1 to stage 2 transitions
        if terminal:
            return
        
        # Increment the count for the observed transition
        self.transition_counts[current_state, action, next_state] += 1
        # Normalize the transition probabilities for the current state-action pair
        total_transitions = self.transition_counts[current_state, action, :].sum()

        # incremental update
        self.transition_model[current_state, action, :] = self.transition_counts[
                                                          current_state, action,
                                                          :] / total_transitions
        
                
        # TODO integrate high-low update below
        # # high-low update
        # p_high = 0.7
        # # self.transition_model[current_state, action, 0] = 0
        # # what state got vitited more sofar
        # most_visited_state = self.transition_counts[current_state, action, 1] >= total_transitions / 2  
        # most_visited_state = int(most_visited_state)
        # self.transition_model[current_state, action, 1:][1-most_visited_state] = p_high
        # self.transition_model[current_state, action, 1:][most_visited_state] = 1 - p_high  


    def update_q_table(self, state, action, reward, next_state, terminal):
        # Update Q-table using the transition model
        if terminal:  # -> second stage -> update with TD
            self.q_table[state, action] += self.alpha * self.reward_prediction_error(
                state, action, reward, next_state, terminal)

        else:
            # self.q_table[state, action] +=  self.transition_model[state, action, next_state] * self.alpha * self.reward_prediction_error(state, action, reward, next_state, terminal)
            # self.q_table[state, action] =  self.transition_model[state, action, next_state] * self.alpha * self.reward_prediction_error(state, action, reward, next_state, terminal)

            self.q_table[state, action] = np.sum(
                [self.transition_model[state, action, possible_state] * np.max(
                    [self.q_table[
                         possible_state, action] + self.alpha * self.reward_prediction_error(
                        state, action, reward, next_state, terminal) for action in
                     self.action_space]
                ) for possible_state in self.state_space])

    def reward_prediction_error(self, state, action, reward, next_state, terminal):
        if terminal:
            return reward - self.q_table[state, action]

        next_action = self.policy(next_state)
        return reward + self.gamma * self.q_table[next_state, next_action] - \
            self.q_table[state, action]

    def update_beliefs(self, state, action, reward, next_state, terminal):
        self.update_transition_model(state, action, next_state, terminal)
        self.update_q_table(state, action, reward, next_state, terminal)

    def reset(self):
        pass
