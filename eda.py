import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from classify import crow_distance

def read_in():
    '''
    OUTPUT
        - pandas dataframe from the property_listings.csv
        - pandas dataframe from the northva-properties-cleaned.csv

    Original digestion of csv tables
    '''

    properties = '../data/denver-properties-in-super-zones.csv'
    return (pd.read_csv(properties, low_memory=False))

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

    df_inds = df.set_index('id')

    df_run = df_inds.drop(['zip', 'year_built'], 1)

    # for i_out, v_out in df_clean_run.groupby('super_zone_id').agg('count')['zone_id'][df_clean_run.groupby('super_zone_id').agg('count')['zone_id'] < 16].iteritems():
    #     super_zone_center = (np.average(df_clean_run[df_clean_run['super_zone_id'] == i_out]['lat']), np.average(df_clean_run[df_clean_run['super_zone_id'] == i_out]['lng']))
    #     for i_in, v_in in df_clean_run[df_clean_run['super_zone_id'] =! i].iteritems():

    sz_df = df_clean_run.groupby('super_zone_id').agg({'zone_id':['count'], 'lat':['mean'], 'lng':['mean']})
    sz_df.columns = sz_df.columns.get_level_values(0)

    underpop_limit = 15
    sz_df['true_zone'] = 0
    for idx_underpop, row_underpop in sz_df[sz_df['zone_id'] < (underpop_limit + 1)].iteritems():
        for idx_full, row_full in sz_df[sz_df['zone_id'] > underpop_limit].iteritems():
        dist = crow_distance(row_underpop, row_full)


    return (df_inds, df_run)

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

    df_clean, df_clean_run = eng_features(df_clean_small)
    print 'extra features added...'

    # df_clean_test = partition_df(df_clean)
    # print 'dataframe partitioned...'

    # return (df_clean, df_clean_run)
    return (df_clean, df_clean_run)
    print 'done!'

if __name__ == '__main__':
    df_clean, df_clean_run = eda_main()
