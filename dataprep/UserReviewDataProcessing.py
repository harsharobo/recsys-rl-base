import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm
from utils.utils import write_rest_index_data, write_user_index_data


def read_data(review_file_path, path):
    rest_reviews = pd.read_csv(review_file_path)
    print(rest_reviews.columns.values)
    print(rest_reviews.shape)
    rest_reviews['date'] = pd.to_datetime(rest_reviews['date'].str.strip(), errors='coerce', format='%Y-%m-%d %H:%M:%S')

    unique_restids = set(rest_reviews.business_id.values)
    rest_idx_dict = {}
    count = 0
    for restid in tqdm(unique_restids):
        rest_idx_dict[restid] = count
        count += 1
    rest_reviews['rest_idx'] = rest_reviews.apply(lambda row: rest_idx_dict[row.business_id], axis=1)

    write_rest_index_data(rest_idx_dict)

    unique_userids = set(rest_reviews.user_id.values)
    user_idx_dict = {}
    count = 0
    for userid in tqdm(unique_userids):
        user_idx_dict[userid] = count
        count += 1

    write_user_index_data(user_idx_dict)

    rest_reviews['user_idx'] = rest_reviews.apply(lambda row: user_idx_dict[row.user_id], axis=1)
    rest_reviews.to_csv(path+'user_rest_reviews_normed.csv', index=False)
    print('user reviews data saved to --> ', path+'user_rest_reviews_normed.csv')
    return rest_reviews


def plot_user_review_dist(reviews_plot_data):
    reviews_plot_data = reviews_plot_data.groupby(['user_id'])['date'].count().reset_index()
    reviews_plot_data.columns = ['user_id', 'review_count']
    # fig, ax = plt.subplots()
    # reviews_plot_data.review_count.hist(ax=ax,bins=250, bottom=0.1)
    # ax.set_yscale('log')
    # plt.show()

    temp_users_list = list(reviews_plot_data[
        (reviews_plot_data.review_count >= 10) & (reviews_plot_data.review_count <= 300)].user_id.unique())
    return temp_users_list


def plot_rating_distribution(reviews_plot_data):
    ax = sns.countplot(reviews_plot_data['stars'])
    plt.show()


def get_train_test_data(reviews_data, user_list, path, k_value, seed):
    shuffled_user_list = shuffle(user_list, random_state=seed)
    cutoff = int(0.85 * len(shuffled_user_list))
    train_users = shuffled_user_list[:10]
    test_users = shuffled_user_list[cutoff:]
    print(len(train_users))
    print(len(test_users))

    reviews_data = reviews_data[reviews_data.stars >= 3.0]
    reviews_data = reviews_data[reviews_data.user_id.isin(user_list)]
    reviews_data = reviews_data.sort_values(['user_id', 'date'])

    train_users_df = reviews_data[reviews_data.user_id.isin(train_users)].copy()
    train_indices = train_users_df.groupby(['user_id']).head(k_value).index
    test_users_df = reviews_data[reviews_data.user_id.isin(test_users)].copy()
    test_indices = test_users_df.groupby(['user_id']).head(k_value).index

    train_history = train_users_df.loc[train_users_df.index.isin(train_indices)]
    train_validation = train_users_df.loc[~train_users_df.index.isin(train_indices)]
    test_history = test_users_df.loc[test_users_df.index.isin(test_indices)]
    test_validation = test_users_df.loc[~test_users_df.index.isin(test_indices)]

    print('train history shape --', train_history.shape)
    print('train validation shape --', train_validation.shape)
    print('test history shape --', test_history.shape)
    print('test validation shape --', test_validation.shape)

    train_history.to_csv(path + 'train_history.csv', index=False)
    train_validation.to_csv(path + 'train_validation.csv', index=False)
    test_history.to_csv(path + 'test_history.csv', index=False)
    train_history.to_csv(path + 'test_validation.csv', index=False)

    print('train history path --', path + 'train_history.csv')
    print('train validation shape --', path + 'train_validation.csv')
    print('test history shape --', path + 'test_history.csv')
    print('test validation shape --', path + 'test_validation.csv')

    return train_history, train_validation, test_history, test_validation


if __name__ == '__main__':
    BASE_PATH = 'D:/Learning/LJMU-masters/recommender_system/workspace/rest_procssed_data/'
    SEED = 123456
    K = 5

    rest_reviews_df = read_data(BASE_PATH + 'filt_rest_review_data.tar.gz')
    print(rest_reviews_df.head().to_string())

    plot_rating_distribution(rest_reviews_df)
    users_list = plot_user_review_dist(rest_reviews_df)

    train_hist_df, train_val_df, test_hist_df, test_val_df = \
        get_train_test_data(rest_reviews_df, users_list, K, SEED)

    print(train_hist_df.head().to_string())
