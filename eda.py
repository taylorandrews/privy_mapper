import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from math import sin, cos, sqrt, atan2, radians


def read_in():
    '''
    OUTPUT
        - pandas dataframe from the property_listings.csv
        - pandas dataframe from the northva-properties-cleaned.csv

    Original digestion of csv tables
    '''

    properties = '../data/denver-properties-in-super-zones.csv'
    return (pd.read_csv(properties, low_memory=False))

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

def clean_df(df, data_types={}):
    '''
    INPUT:
        - dataframe to be cleaned
        - dictionary of column names as keys and desired datatypes as corresponding values

    OUTPUT:
        - dataframe

    Cleans the dataframe.
    Drops columns not listed in the given data_types dictionary.
    Will break if there are NaNs.
    Streamlines datatypes within each column.
    '''

    cols_to_drop = []
    for col in df.columns:
        if col in data_types:
            if data_types[col] == 'datetime64[ns]':
                df.loc[:,col] = pd.to_datetime(df.loc[:,col])
            elif col == 'zip':
                df.loc[:,col] = df.loc[:,col].apply(lambda x: int(x[:5]))
            else:
                df.loc[:,col] = df.loc[:,col].astype(data_types[col])
        else:
            cols_to_drop.append(col)
    df.drop(cols_to_drop, axis=1, inplace=True)

    return df

def eng_features(df):
    '''
    INPUT:
        - dataframe

    OUTPUT
        - dataframe

    Place in algorithm to engineer features

    Also takes any super zones with fewer than 26 houses and lumps them in with the closest zone
    '''

    df_run = df.set_index('id')

    sz_df = df_run.groupby('super_zone_id').agg({'zone_id':['count'], 'lat':['mean'], 'lng':['mean']})
    sz_df.columns = sz_df.columns.get_level_values(0)

    house_limit = 15
    min_dist = 1000
    for idx_underpop, row_underpop in sz_df[sz_df['zone_id'] < (house_limit + 1)].iterrows():
        for idx_full, row_full in sz_df[sz_df['zone_id'] != idx_underpop].iterrows():
            dist = crow_distance(row_underpop, row_full)
            if dist < min_dist:
                min_dist = dist
                nearest_super_zone = idx_full
        # go back through the main df houses in an underpopulated super zone and change their zone to the nearest super zone
        df_run.loc[df_run['super_zone_id'] == idx_underpop, 'super_zone_id'] = nearest_super_zone
    return (df_run)

def partition_df(df):
    '''
    INPUT
        - dataframe

    OUTPUT
        - dataframe that is a subset of input

    This grabs a piece of the dataframe for faster testing
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
    if len(cols) > 0:
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
    print 'data types streamlined...'

    df_clean = eng_features(df_clean_small)
    print 'extra features added...'

    # df_clean_test = partition_df(df_clean)
    # print 'dataframe partitioned...'

    # return (df_clean)
    return (df_clean.loc[df_clean['super_zone_id'].isin([5, 19, 52])])
    print 'done!'

# if __name__ == '__main__':
#     df_clean = eda_main()
