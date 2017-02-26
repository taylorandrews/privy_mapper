import numpy as np
import pandas as pd
from entropy import entropy_main
from eda import eda_main
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

def distance(h1, h2):
    '''
    INPUT
        - row from pandas dataframe representing one house
        - row from pandas dataframe representing another house

    OUTPUT
        - sudo distance between the houses

    Returns how 'far apart' two houses are based on physical distance as well as other feature differences
    '''
    h1_lat, h1_lng, h2_lat, h2_lng = float(h1[11]), float(h1[12]), float(h2[11]), float(h2[12])

    # this is just the pure distance for now
    d = np.sqrt((h2_lat-h1_lat)**2+(h2_lng-h1_lng)**2)
    return d

def potential_web(h, df):
    '''
    INPUT
        - row from pandas dataframe representing one house
        - full dataframe of houses

    OUTPUT
        - subset of dataframe with all potential web partners

    Returns dataframe with all potential web partners based on:
        - same zipcode
        - not already in same web
        - not itself
    '''

    # only keeps houses from the same zipcode
    df = df[df['zip'] == h['zip']]

    # kicks out houses that are already in the same web
    # iff the house in question is already in a web
    if h['web_id'] != 0:
        same_web_mask = df['web_id'].isin(h['web_id'])
        df = df[~same_web_mask]

    # kicks out the house itself from it's own potential comps
    self_mask = df.index.isin(h.index)
    df[~self_mask]

    return df

def join_houses(df):
    '''
    INPUT
        -

    OUTPUT
        -

    Pick a random house and joins the "nearest" one into it's web
    '''

    # finding set of houses in potential web partners
    h = df.sample()

    df_potential_web = potential_web(h, df)

    d_ = 100
    for house in df_potential_web:
        d = distance(h, house)
        if d < d_:
            d_ = d
            if h.index == 0:

                df.set_value(h.index, 'x', 10)
                df.set_value('C', 'x', 10)
            else:
                pass

def build_connectivity_matrix(X, n_neighbors=20):
    '''
    INPUT
        - dataframe to transform into a connectivity matrix that will be used to train the agglomerative clustering model [n_samples, n_features]

    OUTPUT
        - sparse matrix [n_samples, n_samples]

    builds a connectivity matrix that is the 'distance' from each house to every other house in the data
    references the distance function in this file
    '''

    # X = df.values
    connectivity_matrix = kneighbors_graph(X=X, n_neighbors=n_neighbors, mode='distance', metric=distance, include_self=False, n_jobs=-1)

    return connectivity_matrix

def min_entropy_scorer(estimator, X):
    y_pred = estimator.fit_predict(X)
    score = total_entropy(X)


def build_model(X):#, param_grid):
    '''
    INPUT
        -

    OUTPUT
        -

    wuditdo
    '''

    connectivity_matrix = build_connectivity_matrix(X)

    # this one will be used for the grid search
    # ac = AgglomerativeClustering(memory='cluster_cache', compute_full_tree=True)


    # this one is for testing
    ac = AgglomerativeClustering(n_clusters=15,
                                connectivity=connectivity_matrix,
                                affinity='euclidean',
                                linkage='ward')

    y = ac.fit_predict(X)

    # ac.fit()

    # classifier = GridSearchCV(ac, param_grid, scoring=min_entropy_scorer, n_jobs=-1, verbose=1)



    return y

def numpy_to_pandas(X, y, X_cols, y_col='nest_id'):
    '''
    INPUT
        - data to go into dataframe [n_samples, n_features]
        - data to add onto the last column of the dataframe [nsamles]
        - columns headers for dataframe [n_features]
        - column header for the y values to be added on [one_item]

    OUTPUT
        - reindexed pandas dataframe

    Takes multidimentional numpy array and adds on a y column
    '''

    labeled_data = np.vstack((X.T, y)).T
    X_cols.append(y_col)
    df = pd.DataFrame(data=labeled_data, columns=X_cols)

    return df

if __name__ == '__main__':
    # df_clean, entropies = entropy_main()

    ##### ----- TESTING ----- #####
    df_clean = eda_main()
    df_test = df_clean[df_clean['zip'] == 22181]
    cols = list(df_test.columns)
    X = df_test.values
    y = build_model(X)


    ##### ----- RUNNING ----- #####
    # df_clean = eda_main()
    # cols = list(df_clean.columns)
    # X = df_clean.values
    # param_grid= {n_clusters=[10],
    # connectivity=cm,
    # affinity='euclidean',
    # linkage='ward'}
    # y = build_model(X, param_grid)

    # cm = build_connectivity_matrix(df_test)

    df_nest = numpy_to_pandas(X, y, cols)
    entropy_nest = entropy_main(df_nest, 'nest_id')
    entropy_native = entropy_main(df_test, 'zone_id')
