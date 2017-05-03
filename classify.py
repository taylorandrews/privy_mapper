import numpy as np
import pandas as pd
from eda import eda_main, standerdize_cols
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
from math import sin, cos, sqrt, atan2, radians
from sklearn.metrics import silhouette_score, silhouette_samples
from collections import Counter

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

    connectivity_matrix = build_connectivity_matrix(X, urban)

    ac = AgglomerativeClustering(n_clusters=n_clusters,
                                connectivity=connectivity_matrix,
                                affinity='euclidean',
                                linkage='ward'
                                )

    y = ac.fit_predict(X)

    # print y
    # print X
    # sets up a dictionary to store key=cluster id: value=numpy array -> [avg lat, avg lng]
    centroids = {}
    for cluster_id in np.unique(y):
        centroids[cluster_id] = X[np.array([y[i] == cluster_id for i in range(len(y))]).T].mean(0)
    cluster_sizes = Counter(y)


    reassign = []
    min_dist = 1000
    min_cluster_size = 20
    [reassign.append(j) for j in [np.argwhere(i==y) for i in np.unique(y)] if len(j)<min_cluster_size] # builds a list for the current super zone of the house ids that are in a cluster by themselves
    if len(reassign) != 0:
        for (solo_pred, solo_idx) in [(y[i], i) for i in np.concatenate(reassign).ravel()]:
            for cent in [k for (k, v) in cluster_sizes.items() if v > (min_cluster_size - 1)]:
                dist = my_distance(centroids[cent], X[solo_idx], urban)
                if dist < min_dist:
                    min_dist = dist
                    y[solo_idx] = cent
    return y

def numpy_to_pandas(X, y, X_cols, y_col, inds):
    '''
    INPUT
        - data to go into dataframe [n_samples, n_features]
        - data to add onto the last column of the dataframe [n_samples]
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
    df.loc[:,y_col] = df.loc[:,y_col].astype(int)
    return df

def starters(df_clean):
    '''
    INPUT
        - full pandas dataframe

    OUTPUT
        - dictionary where keys are unique zip codes from the input df
        - manual list of columns that need standardizing later on

    gets things started by defining zip codes to loop though and columns to standardize
    '''

    cols_std = []
    df_agg_zips = df_clean.groupby('super_zone_id').agg(lambda x: x.value_counts().index[0]) # creates dataframe with super_zone_id as index, zip as only column
    df_zip_codes = pd.read_csv('../data/zip_codes.csv') # pulls in zip code info for every zip in the US. column 'lzden' is population density
    df_agg_zips_info = pd.merge(df_agg_zips, df_zip_codes, on='zip', how='left') #adds zip code info for each super zone to the agg dataframe.
    super_zone_zip = dict(zip(df_agg_zips.index.values, df_agg_zips.zip)) # creates a dictionary with super zone as key, most common zip as value
    df_agg_zips_info['urban'] = (df_agg_zips_info['lzden'] > 7.85).astype(int) # creates new binary column 'urban' in the agg dataframe
    zip_codes = df_agg_zips_info.set_index('zip').to_dict()['urban'] # creates dictionary with zip code as key, urban boolean as value
    super_zone_dict = {super_zone: (super_zone_zip[super_zone], zip_codes[super_zone_zip[super_zone]]) for super_zone in super_zone_zip} # creates new (and final) dictionary with super zone id as key, (most common zip, urban boolean) tuple as value

    return super_zone_dict, cols_std

def classify_sz(df_sz, urban, cols_std, sz_key):
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

    df = standerdize_cols(df_sz, cols_std)
    inds = df.index

    X_full = df.values
    ll_cols = [1, 2]
    # cols_dict = {0: [5], 1: [14]}
    cols = ll_cols #+ cols_dict[urban]
    col_names = list(df.columns[cols])
    X = X_full[:,cols]
    if urban == 1:
        n_clusters = df.shape[0]/600+1
    else:
        n_clusters = df.shape[0]/400+1
    y_privy = list(df['zone_id'])
    if n_clusters == 1: # this is for super zones with fewer than 40 houses. it doesn't even mess with any of the clustering code, it just throws them into the same group and calls it good
        y_pred = np.zeros(df.shape[0])
        df_nest = numpy_to_pandas(X, y_pred, col_names, 'nest_id', inds)
        df_nest['zone_id'] = y_privy
        df_nest['nest_score'] = np.zeros(df.shape[0], dtype=int)
        df_nest['zone_score'] = 0
    else:
        y_pred = build_fit_predict(X, n_clusters, urban)
        df_nest = numpy_to_pandas(X, y_pred, col_names, 'nest_id', inds)
        df_nest['zone_id'] = y_privy
        kwds = {'urban': urban}
        if len(set(y_pred)) == 1:
            df_nest['nest_score'] = 0
        else:
            df_nest['nest_score'] = silhouette_samples(X=X, labels=y_pred, metric=my_distance, **kwds)
        if len(set(y_privy)) == 1:
            df_nest['zone_score'] = 0
        else:
            df_nest['zone_score'] = silhouette_samples(X=X, labels=y_privy, metric=my_distance, **kwds)
    return df_nest

def final_wash_save(df_scores, df_clean):
    '''
    INPUT
        - dataframe with my nests, their scores and privys zone scores
        - main df with all the house info

    OUTPUT
        - clean, full dataframe with all the house info, nest/zone/super zone id's and the scores
        - df with just house id's and thier nest id's

    This is the last function that puts the final touches on the df's and saves them to csv's to be used! yay
    '''

    df_scores = df_scores.dropna()
    df_clean_scores = df_clean.join(df_scores, how='left')
    df_clean_scores['cluster_id'] = (df_clean_scores['super_zone_id'].astype(str) + '-' + df_clean_scores['nest_id'].astype(str))
    df_clean_scores['cluster_id_int'] = df_clean_scores['cluster_id'].astype('category')
    df_clean_scores['cluster_id_int'] = df_clean_scores['cluster_id_int'].cat.codes
    df_simple = df_clean_scores[['cluster_id_int']]

    df_clean_scores.to_csv('../results/denver_zones_400_600.csv')
    df_simple.to_csv('../results/denver_zones_simple_400_600.csv')
    return df_clean_scores, df_simple

if __name__ == '__main__':
    df_clean = eda_main()
    sz_dict, cols_std = starters(df_clean)
    df_scores = pd.DataFrame()
    i=len(sz_dict)
    for sz_key, urban in sz_dict.iteritems():
        print "working on super zone id {}. {} more to go.".format(sz_key, i-1)
        i-=1
        df_sz = df_clean[df_clean['super_zone_id'] == sz_key]
        df_sz_run = df_sz.drop(['super_zone_id', 'zip', 'year_built'], 1)
        df_nest = classify_sz(df_sz_run, urban[1], cols_std, sz_key)
        df_scores = df_scores.append(df_nest[['nest_id', 'nest_score', 'zone_score']])
    df_clean_scores, df_simple = final_wash_save(df_scores, df_clean)
