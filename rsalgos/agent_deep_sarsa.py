from collections import deque
import random
import numpy as np

from keras.layers import Input, GRU, Concatenate, Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam

from rsalgos.exploration_exploitation import NearestNeighbourEpsilon


class DeepSARSAAgent(object):

    def __init__(self, state_size, context_feat_dim, recom_length, hyper_param, lr):
        self.context_feat_dim = context_feat_dim
        self.state_size = state_size

        # These are hyper parameters for the DQN
        self.gamma = 0.8  # discount rate
        self.learning_rate = lr
        self.batch_size = 1

        # create replay memory using deque
        self.memory = deque(maxlen=1)

        # create main model and target model
        self.model = self.build_model()
        # self.load_model()

        self.exploration_policy = NearestNeighbourEpsilon(state_size, context_feat_dim, recom_length,hyper_param)
        # self.exploration_policy = EpsilonGreedy()
        # [get_rest_by_id(str(each)) for each in self.restaurant_indices]

    def get_action(self, current_state):
        return self.exploration_policy.get_action(current_state, self.model)

    def train_model(self):
        curr_state, action, reward, next_state, done = self.memory.pop()
        next_action, _ = self.get_action(next_state)

        # print('training for state {} action {} and reward {}'.format(curr_state.shape, action.shape, reward))
        curr_state = np.expand_dims(curr_state, axis=0)  # curr_state.reshape(1, self.state_size, self.context_feat_dim)
        action = np.expand_dims(action, axis=0)  # action.reshape(1, self.context_feat_dim)
        next_state = np.expand_dims(next_state, axis=0)  # next_state.reshape(1, self.state_size, self.context_feat_dim)
        next_action = np.expand_dims(next_action, axis=0)  # next_action.reshape(1, self.context_feat_dim)
        # 3. Update your 'update_output' and 'update_input' batch
        # if done:
        #     target[0] = reward
        # else:
        target_next = self.model.predict([next_state, next_action])
        target = reward + (self.gamma * target_next)

        # print('shape of state {} action {} and target {}'.format(curr_state.shape, action.shape, target))
        # 4. Fit your model and track the loss values
        r = self.model.fit([curr_state, action], target, batch_size=self.batch_size, epochs=1, verbose=0)
        return r.history['loss'][0]

    def append_sample(self, curr_state, action, reward, next_state, done, episode_count):
        # update epsilon value so that we perform exploration w.r.t epochs
        self.exploration_policy.update_epsilon(episode_count)
        if done:
            self.exploration_policy.reset_suggestions()
        # print('state {}, action {}, reward {}, next_state {}'.format(curr_state, action, reward, next_state))
        self.memory.append((curr_state, action, reward, next_state, done))

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
