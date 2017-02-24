import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from datetime import datetime

def fill_df(df, cols_to_drop=[]):
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

    df_score = df.loc[df[split_on] == blob_id]
    df_score_norm = normalize_df(df_score)
    shannon_df = df_score_norm.apply(shannon, axis=0)

    blob_entropy = shannon_df.mean()
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
                    'stories': 'int',
                    'zone_id': 'int'}

    for col in df.columns:
        if data_types[col] != 'datetime64[ns]':
            df.loc[:,col] = df.loc[:,col].astype(data_types[col])
        else:
            df.loc[:,col] = pd.to_datetime(df.loc[:,col])
    return df

def eng_features(df):
    '''
    INPUT:
        - dataframe
            must have columns listed_on and sold_on

    OUTPUT
        - dataframe
            will have columns sold_on and time_on_market

    Engeneers a feature called time_on_market
    Also changes the subdivision column to an integer
    Also changes 'sold_on' column to integer by subtracting the minimum date from each date
    '''
    df['time_on_market'] = df['sold_on'] - df['listed_on']
    df.loc[:,'time_on_market'] = df.loc[:,'time_on_market'].dt.days
    df.drop(['listed_on'], axis=1, inplace=True)

    earliest_date = min(df['sold_on'])
    df['sold_on'] = df['sold_on'] - earliest_date
    df.loc[:,'sold_on'] = df.loc[:,'sold_on'].dt.days

    df[['subdivision']] = df[['subdivision']].apply(lambda x: x.cat.codes)
    return df

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

def read_in():
    '''
    OUTPUT
        - pandas dataframe from the property_listings.csv
        - pandas dataframe from the northva-properties-cleaned.csv

    Original digestion of csv tables
    '''

    properties = '../privy_private/property_listings.csv'
    zones = '../privy_private/northva-properties-cleaned.csv'
    return (pd.read_csv(properties), pd.read_csv(zones, usecols=['listing_number', 'zone_id']))

def fill_zones(df_zones):
    df_zones_full = fill_df(df_zones)
    df_zones_full = df_zones_full.drop(df_zones_full[df_zones_full.zone_id == 0].index)
    return df_zones_full

def fill_properties(df_properties):
    properties_cols_to_drop = ["id",
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
                            "street",
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

    df_properties_full = fill_df(df_properties, properties_cols_to_drop)
    return df_properties_full

def partition_df(df):
    '''
    INPUT
        - dataframe

    OUTPUT
        - dataframe that is a subset of input

    This grabs a piece of the dataframe for faster testing of the entropy function
    '''
    zips_to_keep = [22180, 22181, 22027, 22043, 22046, 22213, 22205, 22207]
    # zips_to_keep = [22180, 22181]
    df_clean_test = df.loc[df['zip'].isin(zips_to_keep)]
    return df_clean_test

def main():
    # df_properties, df_zones = read_in()
    # print 'CSVs read in...'
    #
    # df_zones_full = fill_zones(df_zones)
    # df_properties_full = fill_properties(df_properties)
    # print 'nans removed...'
    #
    # df_properties_zones_full = fill_df(df_properties_full.set_index('listing_number').join(df_zones_full.set_index('listing_number')))
    # print 'tables joined...'
    #
    # df_properties_zones_full_clean = col_data_types(df_properties_zones_full)
    # print 'data types streamlined...'
    #
    # df_clean = eng_features(df_properties_zones_full_clean)
    # print 'extra features added...'
    #
    # df_clean_test = partition_df(df_clean)
    # print 'dataframe partitioned...'

    split_on = 'zone_id'
    entropies = get_entropy(df_clean, split_on)
    print 'entropies calculated...'

    return (df_clean, df_clean_test, entropies)

if __name__ == '__main__':
    df_clean, df_clean_test, entropies = main()
