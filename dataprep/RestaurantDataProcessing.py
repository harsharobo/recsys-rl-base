from sklearn.decomposition import PCA
from utils.utils import write_rest_data, get_rest_by_id, get_keys
from rsalgos.ressys_nearest_neighbour import RecSysNearestActions
import pandas as pd


def pre_process_restaurant_data(restaurant_feat_file_name, path):
    context_feat_df = pd.read_csv(restaurant_feat_file_name)
    print(context_feat_df.columns.values)
    print(context_feat_df.shape)
    context_feat_df = context_feat_df.set_index('business_id')

    context_feat_df['Alcohol'] = context_feat_df.Alcohol.str.lower()
    context_feat_df['NoiseLevel'] = context_feat_df.NoiseLevel.str.lower()
    context_feat_df['RestaurantsAttire'] = context_feat_df.RestaurantsAttire.str.lower()
    context_feat_df['WiFi'] = context_feat_df.WiFi.str.lower()
    context_feat_df['BikeParking'] = context_feat_df.BikeParking.str.lower()
    context_feat_df['Caters'] = context_feat_df.Caters.str.lower()
    context_feat_df['GoodForKids'] = context_feat_df.GoodForKids.str.lower()
    context_feat_df['HasTV'] = context_feat_df.HasTV.str.lower()
    context_feat_df['OutdoorSeating'] = context_feat_df.OutdoorSeating.str.lower()
    context_feat_df['RestaurantsDelivery'] = context_feat_df.RestaurantsDelivery.str.lower()
    context_feat_df['RestaurantsGoodForGroups'] = context_feat_df.RestaurantsGoodForGroups.str.lower()
    context_feat_df['RestaurantsPriceRange2'] = context_feat_df.RestaurantsPriceRange2.str.lower()
    context_feat_df['RestaurantsReservations'] = context_feat_df.RestaurantsReservations.str.lower()
    context_feat_df['RestaurantsTableService'] = context_feat_df.RestaurantsTableService.str.lower()
    context_feat_df['RestaurantsTakeOut'] = context_feat_df.RestaurantsTakeOut.str.lower()

    func_map = {'true':1,'True':1,'false':0,'False':0,'none':0,'None':0}
    context_feat_df[['Caters','GoodForKids','HasTV','OutdoorSeating','RestaurantsDelivery','RestaurantsGoodForGroups',\
                     'RestaurantsReservations','RestaurantsTableService','RestaurantsTakeOut','romantic','intimate',\
                     'classy','hipster','divey','touristy','trendy','upscale','casual','garage','street','validated',\
                     'lot','valet','dessert','latenight','lunch','dinner','brunch','breakfast','BikeParking',\
                     'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']]\
    = context_feat_df[['Caters','GoodForKids','HasTV','OutdoorSeating','RestaurantsDelivery','RestaurantsGoodForGroups',\
                     'RestaurantsReservations','RestaurantsTableService','RestaurantsTakeOut','romantic','intimate',\
                     'classy','hipster','divey','touristy','trendy','upscale','casual','garage','street','validated',\
                     'lot','valet','dessert','latenight','lunch','dinner','brunch','breakfast','BikeParking',\
                     'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']].replace(func_map)

    context_feat_df['Alcohol'] = context_feat_df.Alcohol.map({'full_bar':2,'beer_and_wine':1,'none':0,})
    context_feat_df['NoiseLevel'] = context_feat_df.NoiseLevel.map({'very_loud':4,'loud':3,'average':2,'quiet':1,'none':0})
    context_feat_df['RestaurantsAttire'] = context_feat_df.RestaurantsAttire.map({'casual':3,'formal':2,'dressy':1,'none':0})
    context_feat_df['WiFi'] = context_feat_df.WiFi.map({'paid':2,'free':1,'none':0,'no':0})
    context_feat_df['RestaurantsPriceRange2'] = context_feat_df.RestaurantsPriceRange2.map({'1':1,'2':2,'3':3,'4':4,'none':0})

    context_feat_df.to_csv(path+'rest_context_numeric_feat.csv')
    print('data of contextual features written to --',path+'rest_context_numeric_feat.csv')
    return context_feat_df


def reduce_dimension(rest_context_feat, features_selected, path, n_components=6):
    features_select_df = rest_context_feat[features_selected]
    pca = PCA(n_components=n_components)
    rest_context_pc = pca.fit_transform(features_select_df)
    print('variance explained by the PCA is --',pca.explained_variance_ratio_)

    rest_context_pc_df = pd.DataFrame(data=rest_context_pc, index=features_select_df.index)
    rest_context_pc_df.to_csv(path + 'rest_context_pca_feat.csv')
    print('data of contextual features written to --', path + 'rest_context_pca_feat.csv')
    #create dump of restaurant contextual information
    write_rest_data(rest_context_pc_df)
    return rest_context_pc_df


SELECTED_RESTAURANT_FEATURES = ['latitude','longitude', 'Alcohol','NoiseLevel',
 'RestaurantsAttire','WiFi','BikeParking','Caters','GoodForKids','HasTV',
 'OutdoorSeating','RestaurantsDelivery','RestaurantsGoodForGroups',
 'RestaurantsPriceRange2' ,'RestaurantsReservations',
 'RestaurantsTableService', 'RestaurantsTakeOut' ,'romantic', 'intimate',
 'classy', 'hipster', 'divey', 'touristy', 'trendy', 'upscale' ,'casual',
 'garage', 'street', 'validated', 'lot', 'valet', 'dessert', 'latenight', 'lunch',
 'dinner' ,'brunch', 'breakfast']

if __name__ == '__main__':
    BASE_PATH = 'D:/Learning/LJMU-masters/recommender_system/workspace/rest_procssed_data/'
    restaurant_feat_file = BASE_PATH + 'rest_context_feat.csv'

    rest_context_feat_df = pre_process_restaurant_data(restaurant_feat_file, BASE_PATH)

    rest_conext_feat_pca_data = reduce_dimension(rest_context_feat_df, SELECTED_RESTAURANT_FEATURES, BASE_PATH)
    print(rest_conext_feat_pca_data.head())

    print('data from cache is -->', get_rest_by_id('QXAEGFB4oINsVuTFxEYKFQ'))
    print('data from cache is -->', get_rest_by_id('QXAEGFB4oINsVuTFxEYKFQ'))
    print('total keys in cache --> ', get_keys())

    nn_class = RecSysNearestActions()
    nn_class.train_nearest_neighbour(rest_conext_feat_pca_data)
    vect = rest_conext_feat_pca_data.loc['QXAEGFB4oINsVuTFxEYKFQ']
    n_recomm_indices = nn_class.get_nearest_neighbours(vect, 50)

    source = rest_context_feat_df.loc['QXAEGFB4oINsVuTFxEYKFQ']
    print(source)

    resp = rest_context_feat_df.iloc[n_recomm_indices]
    print(resp)
