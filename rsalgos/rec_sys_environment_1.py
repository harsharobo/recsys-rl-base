import random
from sklearn.metrics.pairwise import cosine_similarity
from rsalgos.matrix_factorization_cf import MatrixFactorizationKeras
from utils.utils import *


class RecSysEnvironment(object):

    def __init__(self, user_data_history, user_data_validation, k_hist=5):
        self.user_data_history = user_data_history
        self.user_data_validation = user_data_validation
        self.all_users = list(user_data_history['user_id'].unique())

        self.state_dim = k_hist
        self.current_step = 0
        self.similarity_threshold = 0.999
        self.episode_length = 10

        self.curr_user = None
        self.curr_user_hist = None  # current state
        self.curr_user_val = None  # oracle values of the next action of user
        self.is_done = False

        # initialize disk cache for restaurants data - restaurant_context_data
        self.restaurant_indices = get_keys()

        # initialize the model for validating the recommendation, kind of reward function
        # MF, FM, RNN function, List_Wise Recommendation Cosine_Sim function
        self.mf_class = MatrixFactorizationKeras()
        self.mf_class.load_model()
        # self.reset()

    def reset(self, is_train=True):
        self.current_step = 0
        if is_train:
            self.curr_user = random.choice(self.all_users)
        else:
            self.curr_user = self.all_users[self.current_step]
        # self.current_step = self.current_step+1

        self.curr_user_val = self.user_data_validation[self.user_data_validation.user_id == self.curr_user]
        self.curr_user_hist = self.user_data_history[self.user_data_history.user_id == self.curr_user]
        current_state = []
        for each_rest_id in list(self.curr_user_hist.business_id):
            curr_restid_vec = get_rest_by_id(each_rest_id)
            current_state.append(curr_restid_vec)

        current_state = np.asarray(current_state)
        return current_state

    def step(self, current_state, recomm_actions):
        # check if the recomm_actions are valid ones
        for each in recomm_actions:
            assert each in self.restaurant_indices

        next_action_gt = self.curr_user_val.iloc[self.current_step]
        # print('next action ground truth --> ', next_action_gt.to_string())
        next_state, action_reward_array = self.__perform_state_transition(current_state, recomm_actions, next_action_gt)

        is_done = self.__check_is_episode_done()
        return current_state, action_reward_array, next_state, is_done

    def __perform_state_transition(self, current_state, recomm_actions, next_action_gt):
        actual_rating = next_action_gt['stars']
        restaurant_id = next_action_gt['business_id']
        pca_rest_vector = get_rest_by_id(str(restaurant_id))

        action_similarity = self.__get_action_similarity(pca_rest_vector, recomm_actions)
        action_reward_vect = []
        current_state_temp = current_state.tolist()
        if restaurant_id in list(recomm_actions):
            reward = actual_rating
            current_state_temp.append(pca_rest_vector)
            current_state_temp = current_state_temp[1:self.state_dim + 1]
            print('its a match case for rest {} with reward {}'.format(restaurant_id, reward))
            action_reward_vect.append((pca_rest_vector, reward))
        elif np.max(action_similarity) > self.similarity_threshold:
            reward = actual_rating
            temp_rest_id = recomm_actions[np.argmax(action_similarity)]
            temp_pca_rest_vec = get_rest_by_id(str(temp_rest_id))
            current_state_temp.append(temp_pca_rest_vec)
            current_state_temp = current_state_temp[1:self.state_dim + 1]
            print('its closest match case for rest {} with reward {}'.format(temp_rest_id, reward))
            action_reward_vect.append((temp_pca_rest_vec, reward))

        next_state = np.asarray(current_state_temp)
        # print(next_state)
        return next_state, action_reward_vect

    def __get_action_similarity(self, next_pca_rest_vector, recomm_actions):
        # print('ground truth action vector --', pca_rest_vector)

        pca_recomm_mat = [get_rest_by_id(str(each)) for each in recomm_actions]
        simi_score_matrix = cosine_similarity([next_pca_rest_vector], pca_recomm_mat)
        simi_score_matrix = np.reshape(simi_score_matrix, -1)
        # print('sim score matrix --', simi_score_matrix)

        # rest_idxs = [get_rest_index_by_id(str(each)) for each in recomm_actions]
        # curr_user_idx = [get_user_index_by_id(str(self.curr_user))] * len(recomm_actions)
        # rewards_array = self.mf_class.predict_rating(curr_user_idx, rest_idxs) + self.mu
        # print('rewards from MF model --', rewards_array)

        # reward_actions_array = list(zip(rewards_array, simi_score_matrix))
        # print('final rewards matrix ', reward_actions_array)
        return simi_score_matrix

    def __check_is_episode_done(self):
        # if the current step counter value is greater than user next available ground truth
        # print('current step {} and length of current user validation data {}'
        # .format(self.current_step, len(self.curr_user_val.index)))
        # or self.current_step >= self.episode_length
        self.current_step = self.current_step + 1
        return self.current_step >= len(self.curr_user_val.index) or self.current_step >= self.episode_length
