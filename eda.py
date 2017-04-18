import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

def read_in():
    '''
    OUTPUT
        - pandas dataframe from the property_listings.csv
        - pandas dataframe from the northva-properties-cleaned.csv

    Original digestion of csv tables
    '''

    properties = '../data/denver-properties-in-super-zones.csv'
    return (pd.read_csv(properties, low_memory=False))

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

def col_data_types(df, data_types={}):
    '''
    INPUT:
        - dataframe to be cleaned

    OUTPUT:
        - cleaned dataframe

    Returns a dataframe with consistent datatypes in each column
    '''

    for col in df.columns:
        if data_types[col] != 'datetime64[ns]':
            df.loc[:,col] = df.loc[:,col].astype(data_types[col])
        else:
            df.loc[:,col] = pd.to_datetime(df.loc[:,col])
    return df

def clean_df(df, data_types={}):
    '''
    INPUT:
        -

    OUTPUT:
        -

    '''

    cols_to_drop = []
    for col in df.columns:
        if data_types[col]:
            if data_types[col] != 'datetime64[ns]':
                df.loc[:,col] = df.loc[:,col].astype(data_types[col])
            else:
                df.loc[:,col] = pd.to_datetime(df.loc[:,col])
        else:
            cols_to_drop.append(col)
    df.drop(cols_to_drop, axis=1, inplace=True)

    return df

def eng_features(df):
    '''
    INPUT:
        - dataframe
            must have columns listed_on and sold_on

    OUTPUT
        - dataframe
            will have columns sold_on and time_on_market

    Engineers a feature called time_on_market

    Changes the subdivision column to an integer

    Changes 'sold_on' column to integer by subtracting the minimum date from each date

    Drops houses with fewer than 25 comps in the same zip code

    Drops houses in zip codes with < 2 zones

    Sets indecies to 0, 1, ..., n
    '''
    # df['time_on_market'] = df['sold_on'] - df['listed_on']
    # df.loc[:,'time_on_market'] = df.loc[:,'time_on_market'].dt.days
    # df.drop(['listed_on'], axis=1, inplace=True)
    #
    # earliest_date = min(df['sold_on'])
    # df['sold_on'] = df['sold_on'] - earliest_date
    # df.loc[:,'sold_on'] = df.loc[:,'sold_on'].dt.days
    #
    # df[['subdivision']] = df[['subdivision']].apply(lambda x: x.cat.codes)
    #
    # df = df.groupby('zip').filter(lambda x: len(x) > 25)
    #
    # single_zone = []
    # for z in df['zip'].unique():
    #     if len(df[df['zip'] == z]['zone_id'].unique()) == 1:
    #         single_zone.append(z)
    # df = df[~df.zip.isin(single_zone)]
    # df.reset_index(drop=True, inplace=True)

    return df

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

def standerdize_cols(df, cols):
    '''
    INPUT
        - dataframe
        - list of column names to standardize

    OUTPUT
        - dataframe

    standardizes certain columns in dataframe from 0 to 1
    '''

    scalar = MinMaxScaler()
    df[cols] = scalar.fit_transform(df[cols])
    return df

def eda_main():
    df_properties = read_in()
    print 'CSVs read in...'

    data_types = {'id': 'int',
                  'zone_id': 'int',
                  'super_zone_id': 'int',
                  'zip': 'int',
                  'year_built': 'int',
                  'lat': 'float',
                  'lng': 'float'}

    df_clean_small = clean_df(df_properties, data_types)
    print 'nans removed...'

    # df_properties_zones_full_clean = col_data_types(df_properties_zones_full)
    print 'data types streamlined...'

    df_clean = eng_features(df_clean_small)
    print 'extra features added...'

    # cols_std = ['sold_on', 'time_on_market', 'sold_price', 'above_grade_square_feet', 'lot_size_square_feet', 'basement_square_feet']
    # df_clean_std = standerdize_cols(df_clean, cols_std)
    # print 'relevant columns standardized...'

    # df_clean_test = partition_df(df_clean)
    # print 'dataframe partitioned...'

    return (df_clean)
    print 'done!'
