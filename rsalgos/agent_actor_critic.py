import keras.backend as K
import numpy as np
from collections import deque
import tensorflow as tf
from keras.layers import Input, GRU, Concatenate, Dense, Flatten, Dropout, Lambda, GaussianNoise, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
import random
import gc
from rsalgos.exploration_exploitation import ExplorationPolicy


class Actor(object):

    def __init__(self, state_size, context_feat_dim, learning_rate, tau):
        self.state_size = state_size
        self.context_feat_dim = context_feat_dim
        self.tau = tau
        self.lr = learning_rate

        self.actor = self.create_network()
        self.target_actor = self.create_network()
        self.optimizer = self.optimizer()
        self.target_actor.set_weights(self.actor.get_weights())

    def create_network(self):
        state_input = Input(shape=(self.state_size, self.context_feat_dim,), name='state_input')
        state_gru = GRU(32, return_sequences=True, kernel_initializer='random_uniform', name='state_gru')(state_input)
        state_dense = Dense(32, activation='relu', kernel_initializer='random_uniform', name='state_dense')(state_gru)
        state_flatten = Flatten(name='state_flatten')(state_dense)
        state_dense1 = Dense(16, activation='relu', kernel_initializer='random_uniform', name='state_dense1')(state_flatten)
        # batch_norm = BatchNormalization()(state_flatten)
        # drop_out = Dropout(0.25)(batch_norm)
        state_dense2 = Dense(self.context_feat_dim, activation='relu', kernel_initializer='random_uniform',
                             name='state_dense2')(state_dense1)
        out = Lambda(lambda i: i * self.context_feat_dim)(state_dense2)
        model = Model(inputs=state_input, outputs=out)
        print(model.summary())
        return model

    def optimizer(self):
        action_gdts = K.placeholder(shape=(None, self.context_feat_dim))
        params_grad = tf.gradients(self.actor.output, self.actor.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.actor.trainable_weights)
        #     print(list(grads))
        return K.function(inputs=[self.actor.input, action_gdts], outputs=[],
                          updates=[tf.train.AdamOptimizer(self.lr).apply_gradients(grads)][1:])

    def predict(self, state):
        return self.actor.predict(state)

    def target_predict(self, state):
        return self.target_actor.predict(state)

    def train(self, states, grads):
        self.optimizer([states, grads])

    def transfer_weights(self):
        w, target_w = self.actor.get_weights(), self.target_actor.get_weights()
        for i in range(len(w)):
            target_w[i] = self.tau * w[i] + (1 - self.tau) * target_w[i]
        self.target_actor.set_weights(target_w)


class Critic(object):

    def __init__(self, state_size, context_feat_dim, learning_rate, tau):
        self.state_size = state_size
        self.context_feat_dim = context_feat_dim

        self.learning_rate = learning_rate
        self.tau = tau

        self.critic = self.build_model()
        self.target_critic = self.build_model()
        self.target_critic.set_weights(self.critic.get_weights())

        # Function to compute Q-value gradients (Actor Optimization)
        self.action_grads = K.function([self.critic.input[0], self.critic.input[1]],
                                       K.gradients(self.critic.output, [self.critic.input[1]]))

    def build_model(self):
        state_input = Input(shape=(self.state_size, self.context_feat_dim,), name='state_input')
        state_gru = GRU(32, return_sequences=True, kernel_initializer='he_uniform', name='state_gru')(state_input)
        state_dense = Dense(32, activation='relu', kernel_initializer='he_uniform', name='state_dense')(state_gru)
        state_flatten = Flatten(name='state_flatten')(state_dense)

        action_input = Input(shape=(self.context_feat_dim,), name='action_input')
        action_dense = Dense(16, activation='relu', kernel_initializer='he_uniform', name='action_dense')(
            action_input)

        dense_conc = Concatenate()([state_flatten, action_dense])
        hidden_layer = Dense(32, activation='relu', kernel_initializer='he_uniform')(dense_conc)
        drop_out = Dropout(0.25)(hidden_layer)
        out_layer = Dense(1, activation='linear', kernel_initializer='he_uniform')(drop_out)

        model = Model(inputs=[state_input, action_input], outputs=out_layer)
        model.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate),
            metrics=['mse'],
        )
        print(model.summary())
        return model

    def predict(self, state, action):
        return self.critic.predict([state, action])

    def target_predict(self, state, action):
        return self.target_critic.predict([state, action])

    def transfer_weights(self):
        w, target_w = self.critic.get_weights(), self.target_critic.get_weights()
        for i in range(len(w)):
            target_w[i] = self.tau * w[i] + (1 - self.tau) * target_w[i]
        self.target_critic.set_weights(target_w)

    def train_critic(self, states, actions, critic_target, batch_size):
        r = self.critic.fit([states, actions], critic_target, batch_size=batch_size, epochs=1, verbose=0)
        return r.history['loss'][0]

    def gradients(self, states, actions):
        return self.action_grads([states, actions])


class OUNoise:

    def __init__(self, a_dim, mu=0, theta=0.5, sigma=0.2):
        self.a_dim = a_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.a_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.a_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.rand(len(x))
        self.state = x + dx
        return self.state


class DDPGAgent(object):

    def __init__(self, state_size, context_feat_dim):
        self.state_size = state_size
        self.context_feat_dim = context_feat_dim

        # hyper params
        self.actor_lr = 0.01
        self.critic_lr = 0.01
        self.tau = 0.05
        self.gamma = 0.75  # discount factor

        # actor model class
        self.actor_class = Actor(state_size, context_feat_dim, self.actor_lr, self.tau)

        # critic model class
        self.critic_class = Critic(state_size, context_feat_dim, self.critic_lr, self.tau)

        # Noise with mean 0
        self.noise_class = OUNoise(state_size*context_feat_dim)

        # These are hyper parameters for the DQN
        self.memory = deque(maxlen=5000)
        self.batch_size = 64

        # restaurant contextual information
        self.exploration_policy = ExplorationPolicy()

    def train_model(self):
        if len(self.memory) < self.batch_size:
            return None

        # 1.sample mini batch from the buffer
        mini_batch = random.sample(self.memory, self.batch_size)
        state_batch = np.asarray([each[0] for each in mini_batch])
        action_batch = np.asarray([each[1] for each in mini_batch])
        reward_batch = np.asarray([each[2] for each in mini_batch])
        next_state_batch = np.asarray([each[3] for each in mini_batch])
        done_array = [each[4] for each in mini_batch]

        # print('training state {}, action {}, reward {}, next_state {}'.
        #       format(state_batch.shape, action_batch.shape, reward_batch.shape, next_state_batch.shape))

        # 2. get the q value for the next state from target network
        next_action_weights = self.actor_class.target_predict(next_state_batch)
        # print('next actions from target network --', next_action_weights)

        next_action_batch = np.zeros((self.batch_size, self.context_feat_dim))
        for i in range(self.batch_size):
            next_action_batch[i] = self.__compute_utility(next_action_weights[i])

        next_qvalue_batch = self.critic_class.target_predict(next_state_batch, next_action_batch)
        # print('next q values from target network --', next_qvalue_batch)
        critic_targets = np.zeros((self.batch_size,))
        for i in range(self.batch_size):
            critic_targets[i] = reward_batch[i] + self.gamma * next_qvalue_batch[i]

        # 3. update the critic model with action-values for current batch
        train_loss = self.critic_class.train_critic(state_batch, action_batch, critic_targets, self.batch_size)

        # 4. update the actor with the gradients of the critic model w.r.t action
        action_gradients_weights = self.actor_class.predict(state_batch)
        # print('action from the policy network -- {} -- action taken {}'.format(action_gradients_batch, action_batch))
        action_gradients_batch = np.zeros((self.batch_size, self.context_feat_dim))
        for i in range(self.batch_size):
            action_gradients_batch[i] = self.__compute_utility(action_gradients_weights[i])

        actor_target_batch = self.critic_class.gradients(state_batch, action_gradients_batch)
        # print('gradients of critic wrt to action -- {}'.format(actor_target_batch))
        self.actor_class.train(state_batch, actor_target_batch)

        # 5. transfer the weights from target to actual model
        self.actor_class.transfer_weights()
        self.critic_class.transfer_weights()
        return train_loss

    def append_sample(self, curr_state, action, reward, next_state, done, episode_count):
        # self.memory.append((state, action, reward, next_state))
        # print('state {}, action {}, reward {}, next_state {}, done {}'.
        #       format(curr_state, action, reward, next_state, done))
        if done:
            self.restaurant_indices = get_keys().copy()
        self.memory.append((curr_state, action, reward, next_state, done))

    def get_action(self, current_state):
        # add some noise to the action to query the data points
        noise_array = self.noise_class.noise().reshape(self.state_size, self.context_feat_dim)
        query_state = current_state + noise_array
        action_weights = self.actor_class.predict(np.expand_dims(query_state, axis=0))[0]
        # print('weights from the actor --', action_weights)
        action = self.__compute_utility(action_weights)
        # get the q values for the actions
        # print('action select from the compute utility --', action)
        q_value = self.critic_class.predict(np.expand_dims(current_state, axis=0), np.expand_dims(action, axis=0))
        return action.reshape(-1), q_value.reshape(-1)
        # return next_actions, max_q_values

    def __compute_utility(self, action_weights):
        rest_contextual_matrix = np.asarray([get_rest_by_id(each) for each in self.restaurant_indices])
        # max_actions = np.zeros((action_weights.shape[0], self.context_feat_dim))
        # for i, each_action_weight in enumerate(action_weights):
        score = np.dot(rest_contextual_matrix, action_weights)
        # print('scores matrix -- ', score.shape)
        index = np.argmax(score)
        max_actions = rest_contextual_matrix[index]
        self.restaurant_indices.pop(index)
        return max_actions
