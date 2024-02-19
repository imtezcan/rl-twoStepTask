import numpy as np


# environment of the experiment
class TwoStepEnv:
    # TODO
    # - random seed
    action_space = [0, 1]
    state_space = [0, 1, 2]

    def __init__(self):
        self.state = 0
        # self.action_space = [0, 1]
        # self.state_space = [0, 1, 2]
        self.transition_prob = 0.7
        self.reward = 1
        self.terminal = False
        self.info = {}

        # matrix of transition probabilities
        # 0(action left) -> [0(stay in 0), p(go to 1), 1-p(go to 2)]
        # 1(action right) -> [0(stay in 0), 1-p(go to 1), p(go to 2)]
        self.stage_1_transition_matrix = np.array(
            [[0, self.transition_prob, 1 - self.transition_prob],  # action left
             [0, 1 - self.transition_prob, self.transition_prob]])  # action right

        # self.seed = 0
        # np.random.seed(self.seed)
        self.min_reward_prob = 0.25
        self.max_reward_prob = 0.75
        # matrix of reward probabilities
        # 0(state 0) -> [0 (left), 0(right)]
        # 1(state 1) -> [p1 (left), p2(right)]
        # 2(state 2) -> [p3 (left), p4(right)]
        p_1_0 = np.random.uniform(self.min_reward_prob, self.max_reward_prob)
        p_1_1 = np.random.uniform(self.min_reward_prob, self.max_reward_prob)
        p_2_0 = np.random.uniform(self.min_reward_prob, self.max_reward_prob)
        p_2_1 = np.random.uniform(self.min_reward_prob, self.max_reward_prob)
        # p_1_0 = 0.75
        # p_1_1 = 0.75
        # p_2_0 = 0.25
        # p_2_1 = 0.25

        self.reward_prob_matrix = np.array(
            [[0, 0],  # first stage (state 0) for both actions
             [p_1_0, p_1_1],  # second stage (state 1) for both actions
             [p_2_0, p_2_1]])  # second stage (state 2) for both actions

        # 1 -> fixed reward prob.
        # 0 -> reward prob. can be changed a long the trials
        self.fixed_reward_prob_matrix = np.array([[1, 1],
                                                  [0, 0],
                                                  [0, 0]])

    def reset(self):
        self.state = 0
        self.terminal = False
        self.info = {}
        return self.state

    def step(self, action):
        if self.terminal:
            raise ValueError("Episode has already terminated")
        if action not in self.action_space:
            raise ValueError(
                f"The action: {action} is not valid, action space: {self.action_space}")

        # if in stage 1
        if self.state == 0:
            reward = self.reward_function(self.state, action)  # reward will be 0
            self.state = np.random.choice(self.state_space,
                                          p=self.stage_1_transition_matrix[action])

            self.info["common_transition"] = self.is_common_state(self.state, action)
            self.info["state_transition_to"] = self.state
            self.info["stepOneChoice"] = action

        # if in stage 2
        elif self.state in [1, 2]:
            reward = self.reward_function(self.state, action)
            self.terminal = True
            self.info["reward"] = reward > 0
            self.info["stepTwoChoice"] = action
            # [2:] -> take the reward probabilities for the second stage only
            self.info["rewardProbabilities"] = self.reward_prob_matrix.flatten()[2:]


        else:
            raise ValueError(
                f"state:{self.state} is an invalid state, state space: {self.state_space}")

        return self.state, reward, self.terminal, self.info

    def reward_function(self, state, action):
        if action not in self.action_space:
            raise ValueError(
                f"The action: {action} is not valid, action space: {self.action_space}")
        if state not in self.state_space:
            raise ValueError(
                f"state:{state} is an invalid state, state space: {self.state_space}")

        # give a reward according to the probability of getting a reward
        # for the action taken in the state ( state-action pair )
        reward = np.random.uniform() < self.reward_prob_matrix[state][action]
        # scale the reward for a costume reward value equal to self.reward
        # makes no difference in case self.reward = 1
        reward = reward * self.reward
        return reward

    def state_transition_function(self, state, action):
        if action not in self.action_space:
            raise ValueError(
                f"The action: {action} is not valid, action space: {self.action_space}")

        new_state = None
        terminal = False
        if state == 0:
            new_state = np.random.choice(self.state_space,
                                         p=self.stage_1_transition_matrix[action])
        elif state in [1, 2]:
            terminal = True
        else:
            raise ValueError(
                f"state:{state} is an invalid state, state space: {self.state_space}")

        return new_state, terminal

    def is_common_state(self, state, action):
        if action not in self.action_space:
            raise ValueError(
                f"The action: {action} is not valid, action space: {self.action_space}")
        if state not in self.state_space:
            raise ValueError(
                f"state:{state} is an invalid state, state space: {self.state_space}")

        # return self.stage_1_transition_matrix[action, state] >= 0.5
        return self.stage_1_transition_matrix[action, state] == np.max(
            self.stage_1_transition_matrix[action])

    def set_reward_probabilities(self, reward_prob_matrix):
        if reward_prob_matrix.shape != self.reward_prob_matrix.shape:
            raise ValueError(
                f"reward_prob_matrix shape: {reward_prob_matrix.shape} is not valid, shape should be {self.reward_prob_matrix.shape}")
        # clip the reward probabilities to be between min_reward_prob and max_reward_prob
        reward_prob_matrix = np.clip(reward_prob_matrix, self.min_reward_prob,
                                     self.max_reward_prob)

        # update the reward_prob_matrix
        # if the reward_prob_matrix is fixed -> do not update it, else update it with from the new reward_prob_matrix
        self.reward_prob_matrix = np.where(self.fixed_reward_prob_matrix,
                                           self.reward_prob_matrix, reward_prob_matrix)
        return self.reward_prob_matrix

    def set_seed(self, seed):
        pass

    def plot(self):
        pass