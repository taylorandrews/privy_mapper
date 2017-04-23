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
                                linkage='ward'
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
    df_agg_zips_info['urban'] = (df_agg_zips_info['lzden'] > 8.).astype(int) # creates new binary column 'urban' in the agg dataframe
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
    n_clusters = df.shape[0]/40+1
    y_privy = list(df['zone_id'])
    if n_clusters == 1: # this is for super zones with fewer than 40 houses. it doesn't even mess with any of the clustering code, it just throws them into the same group and calls is good
        y_pred = np.zeros(df.shape[0])
        df_nest = numpy_to_pandas(X, y_pred, col_names, 'nest_id', inds)
        df_nest['zone_id'] = y_privy
        df_nest['nest_score'] = np.zeros(df.shape[0])
        df_nest['zone_score'] = 0
    else:
        y_pred = build_fit_predict(X, n_clusters, urban)
        df_nest = numpy_to_pandas(X, y_pred, col_names, 'nest_id', inds)
        df_nest['zone_id'] = y_privy
        kwds = {'urban': urban}
        df_nest['nest_score'] = silhouette_samples(X=X, labels=y_pred, metric=my_distance, **kwds)
        if len(set(y_privy)) == 1:
            df_nest['zone_score'] = 0
        else:
            df_nest['zone_score'] = silhouette_samples(X=X, labels=y_privy, metric=my_distance, **kwds)
    return df_nest

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
    df_scores = df_scores.dropna()
    df_clean_scores = df_clean.join(df_scores, how='left')
    # df_clean_scores['cluster_id'] =

    # trying to get a nice cluster_id column with super zone id then a 0 then the nest id. concatinate strings of ints ... check nest_id being a float first tho


    df_clean_scores.to_csv('../results/testing.csv')
    df_clean_scores_simple = df_clean_scores.drop(['zone_id', 'super_zone_id', 'zip', 'year_built', 'nest_score', 'zone_score'])
    df_clean_scores_simple.to_csv('../results/testing_simple.csv')


    # there are 35 super zones with fewer than 26 houses. this is the code to see them -> df_clean_run.groupby('super_zone_id').agg('count').sort_values('zone_id')['zone_id'][:35]
