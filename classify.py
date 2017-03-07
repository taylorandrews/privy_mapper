import numpy as np
import pandas as pd
from eda import eda_main, standerdize_cols
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
from math import sin, cos, sqrt, atan2, radians
from sklearn.metrics import silhouette_score, silhouette_samples

def crow_distance(h1, h2):
    '''
    INPUT
        - row from pandas dataframe representing one house
        - row from pandas dataframe representing another house

    OUTPUT
        - distance between the houses over the earths surface as the crow flies

    returns how far apart two houses are based on physical distance
    measured in kilometers
    '''
    h1_lat, h1_lng, h2_lat, h2_lng = float(h1[0]), float(h1[1]), float(h2[0]), float(h2[1])

    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(h1_lat)
    lon1 = radians(h1_lng)
    lat2 = radians(h2_lat)
    lon2 = radians(h2_lng)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    cd = R * c

    return cd

def my_distance(h1, h2, urban):
    '''
    INPUT
        - row from pandas dataframe representing one house
        - row from pandas dataframe representing another house

    OUTPUT
        - sudo distance between the houses

    returns how 'far apart' two houses are based on physical distance as well as other feature differences
    '''

    cd = crow_distance(h1, h2)

    bonus = 1
    # h1_feats = h1[2:]
    # h2_feats = h2[2:]
    #
    # if cd > ((urban * -1.5) + 3):
    #     bonus = 10000
    #
    # for i, v in enumerate(h1_feats):
    #     if v == h2_feats[i]:
    #         bonus *= 0.5

    final_distance = cd * bonus

    return final_distance

def build_connectivity_matrix(X, urban, n_neighbors=10):
    '''
    INPUT
        - dataframe to transform into a connectivity matrix that will be used to train the agglomerative clustering model [n_samples, n_features]
        - boolean indicating urban zip code (1) or suburban (0)
        - integer hyperparameter for the kneighbors graph model

    OUTPUT
        - sparse matrix [n_samples, n_samples]

    builds a connectivity matrix that is the 'distance' from each house to every other house in the data
    references the distance function in this file
    '''

    connectivity_matrix = kneighbors_graph(X=X, n_neighbors=n_neighbors, mode='distance', metric=my_distance, include_self=False, n_jobs=-1, metric_params={'urban': urban})

    return connectivity_matrix

def build_fit_predict(X, n_clusters, urban):
    '''
    INPUT
        - uncategorized data as numpy array [n_samples, n_features]
        - integer hyperparameter for the agglomerative clustering model
        - boolean indicating urban zip code (1) or suburban (0)

    OUTPUT
        - category labels for each data point [n_samples]

    builds a connectivity matrix [n_sample, n_samples]
    creates an agglomerative clustering model that is structured by the connectivity matrix
    fits the model and clusters the data into n_clusters groups
    '''

    connectivity_matrix = build_connectivity_matrix(X, urban) #urban difference

    # this one is for testing
    ac = AgglomerativeClustering(n_clusters=n_clusters,
                                connectivity=connectivity_matrix,
                                affinity='euclidean',
                                linkage='ward',
                                )

    y = ac.fit_predict(X)

    centriods = {}
    for cluster_id in np.unique(y):
        centriods[cluster_id] = X[np.array([y[i] == cluster_id for i in range(len(y))]).T].mean(0)

    single = []
    min_dist = 1000
    [single.append(j) for j in [np.argwhere(i==y) for i in np.unique(y)] if len(j)==1]
    for (solo_pred, solo_ind) in [(y[i], i) for i in np.array(single).flatten()]:
        for cent in centriods:
            dist = my_distance(centriods[cent], X[solo_ind], urban)
            if dist < min_dist:
                y[solo_ind] = cent
    return y

def numpy_to_pandas(X, y, X_cols, y_col, inds):
    '''
    INPUT
        - data to go into dataframe [n_samples, n_features]
        - data to add onto the last column of the dataframe [nsamles]
        - columns headers for dataframe [n_features]
        - column header for the y values to be added on [one_item]
        - list of indecies

    OUTPUT
        - reindexed pandas dataframe

    takes multidimentional numpy array and adds on a y column
    saves the resulting array as a pandas dataframe
    '''

    labeled_data = np.vstack((X.T, y)).T
    X_cols.append(y_col)
    df = pd.DataFrame(data=labeled_data, columns=X_cols, index=inds)

    return df

def starters(df_clean):
    '''
    INPUT
        - full pandas dataframe

    OUTPUT
        - dictionary where keys are unique zip codes from the input df
        - manual list of columns that need atandardizing later on

    gets things started by defining zip codes to loop though and columns to standardize
    '''

    cols_std = ['sold_on', 'time_on_market', 'sold_price', 'above_grade_square_feet', 'lot_size_square_feet', 'basement_square_feet']
    my_zips = [20120]
    # my_zips = [22181, 21054, 20601, 21090, 22025, 20001, 20002, 20009, 20011, 20015]
    # my_zips =list(df_clean['zip'].unique())
    df_zip_codes = pd.read_csv('../data/zip_codes.csv')
    df_clean_zips = pd.merge(df_clean, df_zip_codes, on='zip', how='left')
    df_clean_zips['urban'] = (df_clean_zips['lzden'] > 8.).astype(int)
    df_urban_only = df_clean_zips[df_clean_zips['urban'] == 1]
    urban_zips = list(df_urban_only['zip'])
    zip_codes = {key:(1 if key in urban_zips else 0) for key in my_zips}

    return zip_codes, cols_std

def classify_zip(df_zip, urban, cols_std, zip_key):
    '''
    INPUT
        - dataframe with all rows pretaining to a single zip code
        - boolean indicating urban zip code (1) or suburban (0)
        - list of columns to be standardized within zip code
        - integer zip code with houses being classified

    OUTPUT
        -

    classifies the houses within a given zip code into the same number of nests as there are given zones
    '''

    df = standerdize_cols(df_zip, cols_std)
    inds = df.index
    n_clusters = len(df_zip['zone_id'].unique())
    X_full = df.values

    ll_cols = [11, 12]
    # cols_dict = {0: [5], 1: [14]}
    cols = ll_cols #+ cols_dict[urban]
    col_names = list(df.columns[cols])
    X = X_full[:,cols]
    y_pred = build_fit_predict(X, n_clusters, urban)
    y_privy = list(df['zone_id'])
    df_nest = numpy_to_pandas(X, y_pred, col_names, 'nest_id', inds)
    df_nest['zone_id'] = y_privy
    kwds = {'urban': urban}
    df_nest['nest_score'] = silhouette_samples(X=X, labels=y_pred, metric=my_distance, **kwds)
    df_nest['zone_score'] = silhouette_samples(X=X, labels=y_privy, metric=my_distance, **kwds)

    return df_nest

if __name__ == '__main__':
    df_clean = eda_main()
    zip_codes, cols_std = starters(df_clean)
    df_scores = pd.DataFrame()
    i=len(zip_codes)
    for zip_key, urban in zip_codes.iteritems():
        print "working on zip: {}. {} more to go".format(zip_key, i-1)
        i-=1
        df_zip = df_clean[df_clean['zip'] == zip_key]
        df_nest = classify_zip(df_zip, urban, cols_std, zip_key)
        df_scores = df_scores.append(df_nest[['nest_id', 'nest_score', 'zone_score']])

    df_clean_scores = df_clean.join(df_scores, how='left')
    df_clean_scores.to_csv('../data/data/20120_crow.csv')
