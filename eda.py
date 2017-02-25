import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from datetime import datetime

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

    Engineers a feature called time_on_market

    Changes the subdivision column to an integer

    Changes 'sold_on' column to integer by subtracting the minimum date from each date

    Sets indecies to 0, 1, ..., n

    Drops houses with fewer than 25 comps in the same zip code
    '''
    df['time_on_market'] = df['sold_on'] - df['listed_on']
    df.loc[:,'time_on_market'] = df.loc[:,'time_on_market'].dt.days
    df.drop(['listed_on'], axis=1, inplace=True)

    earliest_date = min(df['sold_on'])
    df['sold_on'] = df['sold_on'] - earliest_date
    df.loc[:,'sold_on'] = df.loc[:,'sold_on'].dt.days

    df[['subdivision']] = df[['subdivision']].apply(lambda x: x.cat.codes)

    df = df.groupby('zip').filter(lambda x: len(x) > 25)

    df.reset_index(drop=True, inplace=True)
    
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

def eda_main():
    df_properties, df_zones = read_in()
    print 'CSVs read in...'

    df_zones_full = fill_zones(df_zones)
    df_properties_full = fill_properties(df_properties)
    print 'nans removed...'

    df_properties_zones_full = fill_df(df_properties_full.set_index('listing_number').join(df_zones_full.set_index('listing_number')))
    print 'tables joined...'

    df_properties_zones_full_clean = col_data_types(df_properties_zones_full)
    print 'data types streamlined...'

    df_clean = eng_features(df_properties_zones_full_clean)
    print 'extra features added...'

    # df_clean_test = partition_df(df_clean)
    # print 'dataframe partitioned...'

    return (df_clean)
    print 'done!'
