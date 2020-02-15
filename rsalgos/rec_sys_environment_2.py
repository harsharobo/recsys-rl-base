import random
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class RecSysEnvironment(object):

    def __init__(self, user_data, all_data, k_hist=5, is_train=True):
        self.user_data = user_data
        self.all_users = list(user_data.user_id.unique())
        self.all_data = all_data

        self.state_dim = k_hist
        self.is_train = is_train
        self.episode_length = 25

        # used for checking state and action similarities
        self.alpha = 0.5

        self.curr_user_data = None
        self.curr_user = None
        self.is_done = False
        self.user_index = 0
        self.current_step = 0

        self.avg_state, self.avg_action, self.rewards, self.group_size = None, None, None, None
        # self.reset()

    def reset(self):
        self.current_step = 0
        if self.is_train:
            self.curr_user = str(random.choice(self.all_users))
            self.curr_user_data = self.user_data[self.user_data.user_id == self.curr_user]
            current_state = self.curr_user_data.sample(1).state.values[0]
        else:
            self.curr_user = str(self.all_users[self.user_index])
            self.curr_user_data = self.user_data[self.user_data.user_id == self.curr_user]
            self.episode_length = self.curr_user_data.shape[0]
            current_state = self.curr_user_data.iloc[self.current_step].state
            self.user_index = self.user_index + 1
            # print('current user {} index {} and curr_user_data {}'.format(self.curr_user,
            #                                                               self.user_index,
            #                                                               self.curr_user_data.shape))

        self.avg_state, self.avg_action, self.rewards, self.group_size = self.__get_average_values()
        # print('starting episode for user -- {} with state {}'.format(
        # str(self.curr_user.user_id.unique()), current_state))
        return current_state

    def step(self, current_state, recomm_action):
        action_reward = self.__simulate_reward(current_state, recomm_action)
        next_state = current_state.copy()
        if action_reward >= 3.0:
            recomm_action = recomm_action[np.newaxis, :]
            next_state = np.append(next_state, recomm_action, axis=0)
            next_state = next_state[1:]

        is_done = self.__check_is_episode_done()
        return current_state, action_reward, next_state, is_done

    def __simulate_reward(self, current_state, action):
        # print('simulating reward for current state {} and action {}'.format(current_state.shape, action.shape))
        comp_current_state = current_state.reshape(1, current_state.shape[0] * current_state.shape[1])
        comp_action = action.reshape(1, -1)
        probability = list()
        denominator = 0.
        # change a different way to calculate simulated reward
        # print('simulating reward for current state {} and action {}'
        # .format(comp_current_state.shape, comp_action.shape))
        for s, a in zip(self.avg_state, self.avg_action):
            s = s.reshape(1, s.shape[0] * s.shape[1])
            a = a.reshape(1, -1)
            numerator = (self.alpha * cosine_similarity(s, comp_current_state)) + \
                        ((1 - self.alpha) * cosine_similarity(a, comp_action))
            probability.append(numerator)
            denominator += numerator
        probability /= denominator
        result = self.rewards[int(np.argmax(probability))]

        # TODO use this for list of recommendation
        # for k, reward in enumerate(simulate_rewards.split('|')):
        #     result += np.power(self.sigma, k) * (0 if reward == "show" else 1)
        return result

    def __get_average_values(self):
        rewards = list()
        avg_states = list()
        avg_actions = list()
        group_sizes = list()
        for reward, group_df in self.all_data[self.all_data.user_id == self.curr_user].groupby(['reward']):
            n_size = group_df.shape[0]
            state_values = group_df['state'].values.tolist()
            action_values = group_df['action'].values.tolist()
            # print('reward {} - state values {} action values {}'.format(reward,
            #                                                             np.asarray(state_values).shape,
            #                                                             np.asarray(action_values).shape))
            avg_states.append(
                np.sum(state_values / np.linalg.norm(state_values, 2, axis=1).clip(1e-8)[:, np.newaxis], axis=0) / n_size
            )
            avg_actions.append(
                np.sum(action_values / np.linalg.norm(action_values, 2, axis=1).clip(1e-8)[:, np.newaxis], axis=0) / n_size
            )
            group_sizes.append(n_size)
            rewards.append(reward)
        return avg_states, avg_actions, rewards, group_sizes

    def __check_is_episode_done(self):
        self.current_step = self.current_step + 1
        return self.current_step >= self.episode_length
