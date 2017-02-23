import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from datetime import datetime

def fill_df(df, cols_to_drop):
    '''
    INPUT:
        - Sparse dataframe to be cleaned
        - List of columns to drop

    OUTPUT:
        Dense dataframe

    Cleans the original dataframe. Simply drops certain columns in cols_to_drop and removes rows with any nan values
    '''

    df.drop(cols_to_drop, axis=1, inplace=True)
    df_full = df.dropna()
    return df_full

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
    # print x
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

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

    df_score_numeric = df.select_dtypes(include=['datetime64[ns]', 'int64', 'float64']) #idk if this will work yet!!!


    # numeric_cols = ['list_price', 'sold_price', 'above_grade_square_feet', 'derived_basement_square_feet', 'garages', 'beds', 'baths', 'zip', 'year_built', 'lot_size_square_feet', 'basement_square_feet', 'lat', 'lng', 'is_attached', ]
    # df_score = df.loc[df[split_on] == blob_id]
    # df_score_numeric = df_score.loc[:,numeric_cols]

    df_score_numeric_norm = normalize_df(df_score_numeric)

    sh_df = df_score_numeric_norm.apply(shannon, axis=0)

    blob_entropy = sh_df.mean()
    if blob_entropy > 0.01:
        return blob_entropy
    else:
        pass
        # return 'Error: Add more comps in the {} blob. \nThere is/are only {} data point(s) in that blob'.format(blob_id, df_score.shape[0])

def shannon(col):
    '''
    INPUT:
        - list of values

    OUTPUT:
        - shannon entropy of given values

    Returns the entropy of a list of values. If one or more values is 0, returns 0 since log(0) is undefined.
    '''
    entropy = - sum([ p * new_log(p) / math.log(2.0) for p in col])
    return entropy

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

def col_data_types(df):
    '''
    INPUT:
        - dataframe to be cleaned

    OUTPUT:
        - cleaned dataframe

    Returns a dataframe with consistent datatypes in each column
    '''
    data_types = {'sold_on': 'datetime64[ns]',
                    'listed_on': 'datetime64[ns]',
                    'sold_price': 'int',
                    'above_grade_square_feet': 'int',
                    'garages': 'int',
                    'beds': 'int',
                    'baths': 'int',
                    'subdivision': 'category',
                    'zip': 'int',
                    'year_built': 'int',
                    'lot_size_square_feet': 'int',
                    'basement_square_feet': 'int',
                    'lat': 'float',
                    'lng': 'float',
                    'is_attached': 'int',
                    'stories': 'int'}

    for col in df.columns:
        if data_types[col] != 'datetime64[ns]':
            df.loc[:,col] = df.loc[:,col].astype(data_types[col])
        else:
            df.loc[:,col] = pd.to_datetime(df.loc[:,col])
    return df

if __name__ == '__main__':
    filename = '../privy_private/property_listings.csv'
    df = pd.read_csv(filename)
    cols_to_drop = ["id",
                    "street",
                    "listing_number",
                    "listing_number_previous",
                    "status",
                    "status_changed_on",
                    "contracted_on",
                    "off_market_on",
                    "list_price",
                    "original_list_price",
                    "previous_price",
                    "derived_basement_square_feet",
                    "car_storage",
                    "car_spaces",
                    "area",
                    "city",
                    "state",
                    "property_key",
                    "externally_last_updated_at",
                    "photos",
                    "structural_style",
                    "property_type",
                    "architecture",
                    "lot_size_acres",
                    "basement_finished_status",
                    "basement_finished_pct",
                    "basement_size",
                    "basement_type",
                    "listing_agent_mls_id",
                    "structural_type",
                    "zoned",
                    "listing_agent_mls_id",
                    "structural_type"]

    print 'here'
    df_full = fill_df(df, cols_to_drop)
    print 'here2'
    # df_full.to_csv('property_listings_22181_testing.csv')

    df_clean = col_data_types(df_full)


    # print df_full.info()
    # print df_full.head()

    # split_on = 'zip'
    #
    # ent_dict = {}
    #
    # # print df_full.info()
    #
    # for col in df_full[split_on].unique():
    #     e_score = entropy_score(df_full, split_on, col)
    #     if e_score:
    #         ent_dict[col] = e_score
    #         # print "{0} entropy: {1}\n".format(col, e_score)
