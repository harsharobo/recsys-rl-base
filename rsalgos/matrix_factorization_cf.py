import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Embedding, Dot, Add, Flatten
from keras.models import Model
from keras.models import load_model
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.utils import shuffle

from dataprep.UserReviewDataProcessing import read_data
from utils.utils import average_precision_k


class MatrixFactorizationKeras(object):

    MODEL_SAVE_PATH = \
        'D:/Learning/LJMU-masters/recommender_system/project-code/rsalgos/models/weights-improvement-10-1.52.hdf5'

    def __init__(self, epochs=10):
        self.latent_space = 8  # latent dimensionality
        self.epochs = epochs
        self.reg = 0.  # regularization penalty

        self.user_size = None
        self.items_size = None
        self.model = None

    def build_model(self, user_rating_df):
        self.user_size = user_rating_df.user_idx.max()+1
        self.items_size = user_rating_df.rest_idx.max() + 1

        u = Input(shape=(1,), name='user-input')
        m = Input(shape=(1,), name='rest-input')
        u_embedding = Embedding(self.user_size, self.latent_space, embeddings_regularizer=l2(self.reg), name='user-embedding')(u)  # (N, 1, K)
        m_embedding = Embedding(self.items_size, self.latent_space, embeddings_regularizer=l2(self.reg), name='rest-embedding')(m)  # (N, 1, K)

        u_bias = Embedding(self.user_size, 1, embeddings_regularizer=l2(self.reg), name='user-bias')(u)  # (N, 1, 1)
        m_bias = Embedding(self.items_size, 1, embeddings_regularizer=l2(self.reg), name='rest-bias')(m)  # (N, 1, 1)

        x = Dot(axes=2, name="Dot-Product")([u_embedding, m_embedding])  # (N, 1, 1)
        x = Add(name="Add-Emb-Bias")([x, u_bias, m_bias])
        x = Flatten(name="Flatten-Users")(x)  # (N, 1)

        model = Model(inputs=[u, m], outputs=x)
        model.compile(
            loss='mse',
            # optimizer='adam',
            # optimizer=Adam(lr=0.01),
            optimizer=SGD(lr=0.05, momentum=0.9),
            metrics=['mse'],
        )
        print(model.summary())
        self.model = model

    def train_model(self, df_train, df_test, mu, batch_size=128):
        # checkpoint
        file_path = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        # Fit the model
        r = self.model.fit(
            x=[df_train.user_idx.values, df_train.rest_idx.values],
            y=df_train.stars.values - mu,
            epochs=self.epochs,
            batch_size=batch_size,
            validation_split=0.20,
            callbacks=callbacks_list
            # validation_data=(
            #     [df_test.user_idx.values, df_test.rest_idx.values],
            #     df_test.stars.values - mu
            # )
        )
        # self.model.save(self.MODEL_SAVE_PATH)
        return r

    def predict_rating(self, user_idx, rest_ids):
        predictions = self.model.predict([user_idx, rest_ids])
        predictions = np.reshape(predictions, -1)
        return predictions

    def load_model(self, file_name=MODEL_SAVE_PATH):
        print('loading the saved MF model --', file_name)
        self.model = load_model(file_name)
        self.model.summary()

    @staticmethod
    def create_test_train_data(user_rest_rating):
        mu = user_rest_rating.stars.mean()
        user_rest_rating = shuffle(user_rest_rating)
        cutoff = int(0.8 * len(user_rest_rating))
        df_train = user_rest_rating.iloc[:cutoff]
        df_test = user_rest_rating.iloc[cutoff:]
        return df_train, df_test, mu


if __name__ == '__main__':
    BASE_PATH = 'D:/Learning/LJMU-masters/recommender_system/workspace/rest_procssed_data/'
    rest_reviews_df = read_data(BASE_PATH + 'filt_rest_review_data.tar.gz')
    print(rest_reviews_df.head().to_string())

    mf_class = MatrixFactorizationKeras()
    mf_class.build_model(rest_reviews_df)
    train_df, test_df, mu = mf_class.create_test_train_data(rest_reviews_df)

    rdata = mf_class.train_model(train_df, test_df, mu)

    plt.plot(rdata.history['loss'], label="train loss")
    plt.plot(rdata.history['val_loss'], label="test loss")
    plt.legend()
    plt.show()

    plt.plot(rdata.history['mean_squared_error'], label="train mse")
    plt.plot(rdata.history['val_mean_squared_error'], label="test mse")
    plt.legend()
    plt.show()

    test_df['actual_rating'] = test_df['stars']
    print('mean rating in the data ', mu)
    print('test set shape ', test_df.shape)
    test_preds = mf_class.predict_rating(test_df.user_idx.values, test_df.rest_idx.values)
    test_df['predict_rating'] = test_preds + mu

    grped_data = test_df.sort_values(['user_idx', 'predict_rating'], ascending=False) \
        .groupby(['user_idx']) \
        .apply(lambda x: average_precision_k(x['actual_rating'], x['predict_rating'])) \
        .reset_index()
    grped_data.columns = ['user_idx', 'avgprecision']
    print(grped_data.avgprecision.value_counts())
    print('mean average precision --> ', np.mean(grped_data.avgprecision))
