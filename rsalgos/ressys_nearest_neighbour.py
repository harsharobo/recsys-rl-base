from sklearn.neighbors import NearestNeighbors
from joblib import dump, load


class RecSysNearestActions(object):

    NN_MODEL_SAVE_PATH = 'models/nearest_neighbour.model'

    def __init__(self):
        self.model_nn = None

    def get_nearest_neighbours(self, query_array, k_neighbours):
        distances, indexs = self.model_nn.kneighbors([query_array], k_neighbours)
        indexs = list(indexs.flatten())
        distances = list(distances.flatten())
        return indexs, distances

    def train_nearest_neighbour(self, rest_context_pc, n_neighbors=10):
        self.model_nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree')
        self.model_nn.fit(rest_context_pc)
        dump(self.model_nn, self.NN_MODEL_SAVE_PATH)
        print('model saved to -->', self.NN_MODEL_SAVE_PATH)

    def load_nearest_neighbour(self):
        print('model loaded from  -->', self.NN_MODEL_SAVE_PATH)
        self.model_nn = load(self.NN_MODEL_SAVE_PATH)
        return self.model_nn
