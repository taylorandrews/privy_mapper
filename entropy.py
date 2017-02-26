from eda import eda_main
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing

def get_entropy(df_clean, split_on):
    '''
    INPUT
        - dataframe with all numeric columns. floats and ints okay
        - heading that defines what category to split on to fin entropy_score
            eg. zone_id to find the entropy of each zone

    OUTPUT
        - dictionary with keys as grouped by feature and values as the entropy for that subset of the data

    Splits up the data and find the entropy of each section and returns it into a dictionary
    '''
    ent_dict = {}
    for col in df_clean[split_on].unique():
        e_score = entropy_score(df_clean, split_on, col)
        if e_score:
            ent_dict[col] = e_score
            # print "{0} entropy: {1}\n".format(col, e_score)

    return ent_dict

def entropy_score(df, split_on, blob_id):
    '''
    INPUT:
        - entire dataframe of which one blob entropy will be calculated
        - the column heading of the feature that will be split on
        - the actual blob_id of the blob in question

    OUTPUT:
        - shannon entropy of the blob

    Get the entropy, or randomness associated with the houses in one blob
    '''

    df_score = df.loc[df[split_on] == blob_id]
    df_score_norm = normalize_df(df_score)
    shannon_df = df_score_norm.apply(shannon, axis=0)

    blob_entropy = shannon_df.mean()
    if blob_entropy > 0.01:
        return blob_entropy
    else:
        pass
        # return 'Error: Add more comps in the {} blob. \nThere is/are only {} data point(s) in that blob'.format(blob_id, df_score.shape[0])

def normalize_df(df):
    '''
    INPUT:
        - Dataframe to be normalized

    OUTPUT:
        - Normalized dataframe

    Normalizes a dataframe so the entropy can be calculated
    '''

    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

def shannon(col):
    '''
    INPUT:
        - list of values

    OUTPUT:
        - shannon entropy of given values

    Returns the entropy of a list of values. If one or more values is 0, returns 0 since log(0) is undefined.
    '''

    # entropy = - sum([ p * new_log(p) / math.log(2.0) for p in col]) / len(col)
    # return entropy

    variance = np.var(col)
    return variance



def new_log(x):
    '''
    INPUT:
        - number, float of int

    OUTPUT:
        - number, float

    Calculates the log of a number, but adjusts for 0's and negatives
    '''

    if x == 0:
        return 0
    elif x > 0:
        return math.log(x)
    else:
        return math.log(-x)

def entropy_main(df, split_on):
    # df_clean = eda_main()
    # print 'data imported...'

    # split_on = 'zone_id'
    entropies = get_entropy(df, split_on)
    total_entropy = sum(entropies[d] for d in entropies) / len(entropies)
    # print 'entropies calculated...'

    return -total_entropy
