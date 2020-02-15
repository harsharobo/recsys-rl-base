from collections import deque
import random
import numpy as np

from keras.layers import Input, GRU, Concatenate, Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam

from rsalgos.exploration_exploitation import NearestNeighbourEpsilon, EpsilonGreedy


class DQNAgent(object):

    def __init__(self, state_size, context_feat_dim, recom_length, hyper_param, lr):
        self.context_feat_dim = context_feat_dim
        self.state_size = state_size

        # These are hyper parameters for the DQN
        self.gamma = 0.8  # discount rate
        self.learning_rate = lr
        self.tau = 0.05
        self.batch_size = 128
        self.train_start = 150

        # create replay memory using deque
        self.memory = deque(maxlen=5000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())

        self.exploration_policy = NearestNeighbourEpsilon(state_size, context_feat_dim, recom_length, hyper_param)
        # self.exploration_policy = EpsilonGreedy()
        # [get_rest_by_id(str(each)) for each in self.restaurant_indices]

    def get_action(self, current_state, use_target=False):
        if use_target:
            return self.exploration_policy.get_action(current_state, self.target_model)
        else:
            return self.exploration_policy.get_action(current_state, self.model)

    def train_model(self):
        loss = None
        if len(self.memory) >= self.batch_size:

            mini_batch = random.sample(self.memory, self.batch_size)
            state_batch = np.asarray([each[0] for each in mini_batch])
            action_batch = np.asarray([each[1] for each in mini_batch])
            reward_batch = np.asarray([each[2] for each in mini_batch])
            next_state_batch = np.asarray([each[3] for each in mini_batch])
            done_array = [1-int(each[4]) for each in mini_batch]

            # print('shape of state {} action {} and next_state_batch {}'.format(state_batch, action_batch, next_state_batch))
            next_action_batch = np.zeros((self.batch_size, self.context_feat_dim))
            for i in range(self.batch_size):
                next_action_batch[i], _ = self.get_action(next_state_batch[i], use_target=True)

            # target = self.model.predict([state_batch, action_batch])
            target_next = self.target_model.predict([next_state_batch, next_action_batch])
            reward_batch = reward_batch[:, np.newaxis]
            target_update = reward_batch + (self.gamma * target_next)

            # print('shape of target_next {} reward_batch {} and target_update {}'.format(target_next, reward_batch, target_update))
            # 4. Fit your model and track the loss values
            r = self.model.fit([state_batch, action_batch], target_update, batch_size=self.batch_size, epochs=1, verbose=0)
            loss = r.history['loss']

            self.update_target_weights()
        return loss

    def append_sample(self, curr_state, action, reward, next_state, done, episode_count):
        # update epsilon value so that we perform exploration w.r.t epochs
        self.exploration_policy.update_epsilon(episode_count)
        if done:
            self.exploration_policy.reset_suggestions()
        # print('state {}, action {}, reward {}, next_state {}'.format(curr_state, action, reward, next_state))
        self.memory.append((curr_state, action, reward, next_state, done))

    def update_target_weights(self):
        w, target_w = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(w)):
            target_w[i] = self.tau * w[i] + (1 - self.tau) * target_w[i]
        self.target_model.set_weights(target_w)

    def build_model(self):
        state_input = Input(shape=(self.state_size, self.context_feat_dim,), name='state_input')
        state_gru = GRU(32, return_sequences=True, kernel_initializer='he_uniform', name='state_gru')(state_input)
        state_dense = Dense(32, activation='relu', kernel_initializer='he_uniform', name='state_dense')(state_gru)
        state_flatten = Flatten(name='state_flatten')(state_dense)

        action_input = Input(shape=(self.context_feat_dim,), name='action_input')
        action_dense = Dense(16, activation='relu', kernel_initializer='he_uniform', name='action_dense')(
            action_input)

        dense_conc = Concatenate()([state_flatten, action_dense])
        # bn = BatchNormalization()(state_gru)
        hidden_layer = Dense(32, activation='relu', kernel_initializer='he_uniform')(dense_conc)
        # dp = BatchNormalization()(hidden_layer)
        drop_out = Dropout(0.25)(hidden_layer)
        out_layer = Dense(1, activation='linear', kernel_initializer='he_uniform')(drop_out)

        model = Model(inputs=[state_input, action_input], outputs=out_layer)
        model.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate),
            # optimizer=SGD(lr=self.learning_rate, momentum=0.9),
            metrics=['mse'],
        )
        print(model.summary())
        return model

    def load_model(self):
        weights_path = \
            'D:/Learning/LJMU-masters/recommender_system/project-code/rsalgos/models/dqn-model-reward-prediction.h5'
        self.model.load_weights(weights_path)
        print(self.model.summary())

    def save_model(self, name):
        self.model.save_weights(name)
