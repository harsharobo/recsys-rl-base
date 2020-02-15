import warnings
warnings.filterwarnings('ignore')

import time
import random
from collections import defaultdict
from rsalgos.agent_deep_sarsa import DeepSARSAAgent
from rsalgos.agent_actor_critic import DDPGAgent
from rsalgos.agent_dqn import DQNAgent
from rsalgos.rec_sys_environment_2 import RecSysEnvironment
from utils.utils import *
from config.ConfigDataProvider import config_data


def run_one_episode(env, agent, episode, policy_type, is_train=True):
    curr_state = env.reset()
    # print("starting episode for user {} --
    # with validation size {}".format(env.curr_user, len(env.curr_user_val.index)))
    done = False
    user_data = {'prec_vect': list(), 'reward_vect': list(), 'action': list()}
    maxq_track_array = list()
    train_loss_array = list()
    while not done:
        # 1. Pick epsilon-greedy action from possible actions for the current state
        one_action, max_qvalue = agent.get_action(curr_state)
        user_data['action'].append(one_action)
        maxq_track_array.append(max_qvalue)

        # 2. perform state transition from current state to next state
        current_state, action_reward, next_state, done = env.step(curr_state, one_action)
        if np.array_equal(current_state, next_state):
            user_data['prec_vect'].append(0)
        else:
            user_data['prec_vect'].append(1)

        # 3. Append the experience to the memory
        agent.append_sample(curr_state, one_action, action_reward, next_state, done, episode)

        # 4. Train the model by calling function agent.train_model
        if policy_type == 'onpolicy' and is_train:
            train_loss = agent.train_model()
            if train_loss:
                train_loss_array.append(train_loss)
                Q_LEARNING_TRACKING[episode] = train_loss_array

        # reset the current state to next_state
        curr_state = next_state
        user_data['reward_vect'].append(action_reward)
        if done:
            USER_STATE_REWARDS[str(env.curr_user)].append(user_data)
            EPISODE_TOTAL_REWARDS[episode] = user_data['reward_vect']
            EPISODE_AVG_PRECISION[episode] = user_data['prec_vect']
            Q_VALUE_TRACKING[episode] = maxq_track_array
            # print("completed one episode for user --", env.curr_user)

    # 4. Train the model by calling function agent.train_model
    if policy_type == 'offpolicy' and is_train:
        train_loss = agent.train_model()
        if train_loss:
            train_loss_array.extend(train_loss)
            Q_LEARNING_TRACKING[episode] = train_loss_array


def pre_process_data():
    user_restaurant_rating_file = DATA_BASE_PATH + 'filt_rest_review_data.tar.gz'
    simulation_file = DATA_BASE_PATH + 'dqn_simulation_data.csv'
    restaurant_embeddings_file = DATA_BASE_PATH + 'restaurant_context_embeddings.csv'
    restaurant_feat_file = DATA_BASE_PATH + 'rest_context_feat.csv'

    rest_embeddings_data = pd.read_csv(restaurant_embeddings_file)
    rest_embeddings_data = rest_embeddings_data.set_index('business_id')
    print('shape of restaurant embeddings data -', rest_embeddings_data.shape)
    print('columns of restaurant embeddings data-', rest_embeddings_data.columns.values)

    simulation_data = pd.read_csv(simulation_file)
    print('shape of simulation data -', simulation_data.shape)
    print('columns of simulation data-', simulation_data.columns.values)
    simulation_data['state'] = simulation_data.state. \
        apply(lambda x: np.asarray([each.split('|') for each in x.split(',')], dtype='float32'))
    simulation_data['action'] = simulation_data.action.apply(lambda x: np.asarray(x.split('|'), dtype='float32'))
    simulation_data['reward'] = simulation_data.reward.astype(float)
    # simulation_data['next_state'] = simulation_data.next_state. \
    #     apply(lambda x: np.asarray([each.split('|') for each in x.split(',')], dtype='float32'))

    rest_data = pd.read_csv(restaurant_feat_file)
    print('shape of restaurant data -', rest_data.shape)
    print('columns of restaurant data-', rest_data.columns.values)

    # selected_rest_list = list(rest_data[rest_data.state == 'AZ'].business_id.unique())
    # cache_data = rest_embeddings_data[rest_embeddings_data.index.isin(selected_rest_list)]
    print('size of restaurant embeddings data after state filter --', rest_embeddings_data.shape)
    write_rest_data(rest_embeddings_data)

    del rest_embeddings_data
    del rest_data

    # select_user_list = list(user_rating_data[user_rating_data.business_id.isin(selected_rest_list)].user_id.unique())
    # select_user_list = list(simulation_data[simulation_data.user_id.isin(select_user_list)].user_id.unique())
    select_user_list = list(simulation_data.user_id.unique())
    print('unique users size from user data --', len(select_user_list))
    select_user_list = [random.choice(select_user_list) for _ in range(USER_SAMPLE_SIZE)]
    print('selected users size from user data --', select_user_list)
    simulation_data = simulation_data[simulation_data.user_id.isin(select_user_list)]

    def get_train_sessions(rows, perc=0.8):
        count = int(len(rows) * perc)
        return list(rows.head(count).index)

    def get_test_sessions(rows, perc=0.2):
        count = int(len(rows) * perc)
        return list(rows.tail(count).index)

    grpd_data = simulation_data.groupby(['user_id'])

    train_idxs = grpd_data.apply(get_train_sessions).reset_index()[0].values
    train_idxs = [idx for each in train_idxs for idx in each]
    simulation_data_train = simulation_data[simulation_data.index.isin(train_idxs)]
    print('size of simulation train data', simulation_data_train.shape)

    test_idnx = grpd_data.apply(get_test_sessions).reset_index()[0].values
    test_idnx = [idx for each in test_idnx for idx in each]
    simulation_data_test = simulation_data[simulation_data.index.isin(test_idnx)]
    print('size of simulation test data', simulation_data_test.shape)

    return simulation_data, simulation_data_train, simulation_data_test


def evaluate_rl_agent(a_type, user_data):
    final_df = pd.DataFrame()
    for user, value_dict in user_data.items():
        episode_df = get_evaluation_dataframe(user, value_dict)
        # print('user={} -- MAP={} and nDCG={}'.format(user, episode_df.avg_prec.mean(), episode_df.ndcg.mean()))
        final_df = pd.concat([final_df, episode_df])

    final_df.to_csv((METADATA_BASE_PATH + '{}_{}_user_evaluation_raw.csv').format(a_type, agent_key), index=False)
    final_df.groupby(['user_id']).agg({'avg_prec': 'mean', 'ndcg': 'mean'}).reset_index() \
        .to_csv((METADATA_BASE_PATH + '{}_{}_user_evaluation_metrics.csv').format(a_type, agent_key), index=False)
    map_train = final_df.avg_prec.mean()
    ndcg_train = final_df.ndcg.mean()
    print('Data metrics for shape {} -- MAP={} and nDCG={}'.format(final_df.shape, map_train, ndcg_train))


if __name__ == '__main__':

    DATA_BASE_PATH = 'data/'
    METADATA_BASE_PATH = 'metadata/'
    MODELS_BASE_PATH = 'models/'

    EPISODES = int(config_data.get_config_value('MAIN', 'episode.length'))
    MAX_RECOM_LENGTH = int(config_data.get_config_value('MAIN', 'max.query.size'))
    SEED = int(config_data.get_config_value('MAIN', 'seed'))
    USER_SAMPLE_SIZE = int(config_data.get_config_value('MAIN', 'user.sample.size'))
    RESTAURANT_FEATURES_N_DIM = 10
    K_WINDOW = 5

    sim_all_data, sim_train_data, sim_test_data = pre_process_data()
    env_train = RecSysEnvironment(sim_train_data, sim_all_data, K_WINDOW)

    agents_list = {'dqn': ('offpolicy', DQNAgent(K_WINDOW, RESTAURANT_FEATURES_N_DIM, MAX_RECOM_LENGTH, 0.05, 0.02)),
                   'deep_sarsa': ('onpolicy', DeepSARSAAgent(K_WINDOW, RESTAURANT_FEATURES_N_DIM, MAX_RECOM_LENGTH, 0.05, 0.02)),
                   'ddpg-ac': ('offpolicy', None)}

    # agents_list = {'dqn_01': ('offpolicy', DQNAgent(K_WINDOW, RESTAURANT_FEATURES_N_DIM, MAX_RECOM_LENGTH, 0.1)),
    #                'dqn_05': ('offpolicy', DQNAgent(K_WINDOW, RESTAURANT_FEATURES_N_DIM, MAX_RECOM_LENGTH, 0.5)),
    #                'dqn_001': ('offpolicy', DQNAgent(K_WINDOW, RESTAURANT_FEATURES_N_DIM, MAX_RECOM_LENGTH, 0.01)),
    #                'dqn_005': ('offpolicy', DQNAgent(K_WINDOW, RESTAURANT_FEATURES_N_DIM, MAX_RECOM_LENGTH, 0.05)),
    #                'deep_sarsa_01': ('onpolicy', DeepSARSAAgent(K_WINDOW, RESTAURANT_FEATURES_N_DIM, MAX_RECOM_LENGTH, 0.1)),
    #                'deep_sarsa_05': ('onpolicy', DeepSARSAAgent(K_WINDOW, RESTAURANT_FEATURES_N_DIM, MAX_RECOM_LENGTH, 0.5)),
    #                'deep_sarsa_001': ('onpolicy', DeepSARSAAgent(K_WINDOW, RESTAURANT_FEATURES_N_DIM, MAX_RECOM_LENGTH, 0.01)),
    #                'deep_sarsa_005': ('onpolicy', DeepSARSAAgent(K_WINDOW, RESTAURANT_FEATURES_N_DIM, MAX_RECOM_LENGTH, 0.05))}

    # agents_list = {'deep_sarsa_01': ('onpolicy', DeepSARSAAgent(K_WINDOW, RESTAURANT_FEATURES_N_DIM, MAX_RECOM_LENGTH, 0.05, 0.01)),
    #                'deep_sarsa_005': ('onpolicy', DeepSARSAAgent(K_WINDOW, RESTAURANT_FEATURES_N_DIM, MAX_RECOM_LENGTH, 0.05, 0.05)),
    #                'deep_sarsa_1': ('onpolicy', DeepSARSAAgent(K_WINDOW, RESTAURANT_FEATURES_N_DIM, MAX_RECOM_LENGTH, 0.05, 0.1)),
    #                'deep_sarsa_25': ('onpolicy', DeepSARSAAgent(K_WINDOW, RESTAURANT_FEATURES_N_DIM, MAX_RECOM_LENGTH, 0.05, 0.25))}

    for agent_key, (policy, agent_obj) in agents_list.items():
        print('starting run for {} with policy evaluation type {}'.format(agent_key, policy))
        # Tracking for convergence update them after each episode
        USER_STATE_REWARDS = defaultdict(list)  # user_id -> [[rewards for eps1], [rewards for eps2]....[rewards for eps-n]]
        EPISODE_TOTAL_REWARDS = defaultdict()  # episode_num -> [rewards given across users]
        EPISODE_AVG_PRECISION = defaultdict()  # episode_num -> [avg prec given across users]
        Q_VALUE_TRACKING = defaultdict()
        Q_LEARNING_TRACKING = defaultdict()

        if not agent_obj:
            continue

        start_time = time.time()
        trained_model = None
        for episodei in tqdm(range(EPISODES + 1)):
            run_one_episode(env_train, agent_obj, episodei, policy)

            # save the model
            if episodei % 1000 == 0:
                agent_obj.model.save_weights((MODELS_BASE_PATH+"{}_{}_{}.h5").format(agent_key, policy, episodei))
                trained_model = agent_obj.model

            if episodei % 500 == 0:
                save_obj(USER_STATE_REWARDS,
                         (METADATA_BASE_PATH+'{}_user_state_rewards.pkl').format(agent_key))
                save_obj(EPISODE_TOTAL_REWARDS,
                         (METADATA_BASE_PATH+'{}_episode_total_rewards.pkl').format(agent_key))
                save_obj(Q_LEARNING_TRACKING,
                         (METADATA_BASE_PATH+'{}_train_loss.pkl').format(agent_key))
                save_obj(Q_VALUE_TRACKING,
                         (METADATA_BASE_PATH+'{}_qvalue_tracking.pkl').format(agent_key))
                save_obj(EPISODE_AVG_PRECISION,
                         (METADATA_BASE_PATH + '{}_episode_avg_precision.pkl').format(agent_key))

            if episodei % 500 == 0:
                end_time = time.time()
                print("episode:", episodei, "  memory length:", len(agent_obj.memory), "  epsilon:", agent_obj.exploration_policy.epsilon)
                print('time taken for 1000 episode --> %.2fs' % (end_time - start_time))
                start_time = time.time()

        # perform MAP/nDCG calculation for episodes.
        evaluate_rl_agent('train', USER_STATE_REWARDS)

        # Plot for the graphs.
        plot_from_dict(EPISODE_TOTAL_REWARDS, np.sum,
                       (METADATA_BASE_PATH+'{}_rewards_tracking.png').format(agent_key))
        plot_from_dict(Q_LEARNING_TRACKING, np.mean,
                       (METADATA_BASE_PATH + '{}_qnetwork_loss.png').format(agent_key))
        plot_from_dict(Q_VALUE_TRACKING, np.mean,
                       (METADATA_BASE_PATH + '{}_qvalue_tracking.png').format(agent_key))

        # Tracking for convergence update them after each episode
        USER_STATE_REWARDS = defaultdict(list)  # user_id -> [[rewards for eps1], [rewards for eps2]....[rewards for eps-n]]
        EPISODE_TOTAL_REWARDS = defaultdict()  # episode_num -> [rewards given across users]
        Q_VALUE_TRACKING = defaultdict()
        Q_LEARNING_TRACKING = defaultdict()

        # perform testing part of the code
        env_test = RecSysEnvironment(sim_test_data, sim_all_data, K_WINDOW, False)
        test_user_len = len(sim_test_data.user_id.unique())
        print('starting testing for {} users'.format(test_user_len))
        start_time = time.time()
        for episodej in tqdm(range(test_user_len)):
            run_one_episode(env_test, agent_obj, episodej, policy)

            save_obj(USER_STATE_REWARDS,
                     (METADATA_BASE_PATH + '{}_test_user_state_rewards.pkl').format(agent_key))
            save_obj(EPISODE_TOTAL_REWARDS,
                     (METADATA_BASE_PATH + '{}_test_episode_total_rewards.pkl').format(agent_key))
            save_obj(Q_VALUE_TRACKING,
                     (METADATA_BASE_PATH + '{}_test_qvalue_tracking.pkl').format(agent_key))
            save_obj(EPISODE_AVG_PRECISION,
                     (METADATA_BASE_PATH + '{}_test_episode_avg_precision.pkl').format(agent_key))

        end_time = time.time()
        print('time taken for testing --> %.2fs' % (end_time - start_time))

        # perform MAP/nDCG calculation for episodes.
        evaluate_rl_agent('test', USER_STATE_REWARDS)

    print('complete the run')
