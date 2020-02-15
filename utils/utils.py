import matplotlib
matplotlib.use('Agg')

from diskcache import Cache
from config import ConfigDataProvider
import pylru
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# DISK_CACHE_PATH = 'data/rest_disk_cache/'
# DISK_RESTID_ID_PATH = './data/restid_index_cache/'
# DISK_USERID_ID_PATH = './data/userid_index_cache/'
#
# disk_cache = Cache(DISK_CACHE_PATH, size_limit=int(2**33), disk_min_file_size = 1*1024,
#                         evication_policy = u'least-frequently-used')
# restid_index_cache = Cache(DISK_RESTID_ID_PATH, size_limit=int(2 ** 33), disk_min_file_size=1 * 1024,
#                         evication_policy = u'least-frequently-used')
# userid_index_cache = Cache(DISK_USERID_ID_PATH, size_limit=int(2 ** 33), disk_min_file_size=1 * 1024,
#                         evication_policy = u'least-frequently-used')

# disk_cache = pylru.lrucache(20000)

disk_cache = dict()


def get_rest_by_id(business_id):
    return disk_cache[business_id]


def write_rest_data(rest_context_df):
    for rest_id in tqdm(list(rest_context_df.index)):
        # print('writing {} data to cache {}'.format(rest_id, np.array(rest_context_df.loc[rest_id])))
        disk_cache[str(rest_id)] = np.array(rest_context_df.loc[rest_id])
    print('disk cache created with size -->', len(disk_cache))
    # disk_cache.close()


def get_keys():
    # print('length of keys in cache ', disk_cache.__len__())
    # return list(disk_cache.__iter__())
    return list(disk_cache.keys())


def average_precision_k(actual_list, pred_list, top_k, threshold_val):
    relev_list = actual_list
    if len(actual_list) > top_k:
        relev_list = actual_list[:top_k]

    if len(relev_list) == 0:
        return 0.0

    score = 0.0
    num_hits = 0.0
    for i, actual_rate in enumerate(relev_list):
        if pred_list[i] >= threshold_val:
            num_hits = num_hits + 1.0
            score = score + (num_hits / (i + 1.0))

    avg_prec = score / min(len(relev_list), top_k)
    return avg_prec


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, a, k, method=0):
    dcg_max = dcg_at_k(a, k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def save_obj(input, path_file_name):
    with open(path_file_name, 'wb') as fp:
        pickle.dump(input, fp)


def load_obj(input_file):
    with (open(input_file, "rb")) as openfile:
        loaded_data = pickle.load(openfile)
        return loaded_data


def get_evaluation_dataframe(user_id, user_data_dict):
    data_df = pd.DataFrame(user_data_dict)
    data_df['user_id'] = user_id
    data_df['ndcg'] = data_df.reward_vect.apply(
        lambda rew: ndcg_at_k(rew, [5] * len(rew), len(rew)))
    data_df['avg_prec'] = data_df.prec_vect.apply(
        lambda pred: average_precision_k([1] * len(pred), pred, len(pred), 1.0))
    return data_df


def plot_from_dict(raw_dict, agg_function, save_path):
    print_list = [(key, agg_function(value)) for key, value in raw_dict.items()]
    x, y = zip(*print_list)
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    fig.savefig(save_path)


def plot_epsilon_hyper_param(epochs):
    x_axis = np.arange(0, epochs)
    epsilon = []
    for i in range(0, epochs):
        epsilon.append(np.exp(-0.0006 * i))

    plt.plot(x_axis, epsilon)
    plt.show()
