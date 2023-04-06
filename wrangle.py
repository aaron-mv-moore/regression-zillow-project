import env
import pandas as pd
import os

import scipy.stats as stats
import numpy as np
from sklearn.model_selection import train_test_split


def get_zillow():
    '''
    Argument: No arguments required
    Actions: 
        1. Checks for the existence of the csv in the current directory
            a. if present:
                i. reads the csv
            b. if not present:
                i. queries MySQL dtabase using the env.py file for the credentials
                ii. saves the csv to the current working directory
    Return: dataframe
    Modules: 
        import env
        import pandas as pd
        import os
    '''
    # a variable to hold the xpected or future file name
    filename = 'zillow.csv'

    # if the file is present in the directory 
    if os.path.isfile(filename):

        # read the csv and assign it to the variable df
        df = pd.read_csv(filename, index_col=0)

        # return the dataframe and exit the funtion
        return df

    # if the file is not in the current working directory,
    else:
        # assign the name of the database to db
        db = 'zillow'

        # use the env.py function to get the url needed from the db
        url = env.get_db_url(db)

        # assign the sql query into the variable query
        query = '''SELECT
                      bedroomcnt,
                      bathroomcnt,
                      calculatedfinishedsquarefeet,
                      taxvaluedollarcnt,
                      fips
                    FROM properties_2017
                      INNER JOIN predictions_2017 ON properties_2017.parcelid = predictions_2017.parcelid
                    WHERE propertylandusetypeID = 261
                      AND YEAR(predictions_2017.transactiondate) = 2017
                    ;'''

        # query sql using pandas function
        df = pd.read_sql(query, url)

        # save the dataframe as a csv to the current working directory
        df.to_csv(filename)

        # returns the dataframe
        return df


def clean_data(df, focus=True):
    '''
    Arguments: zillow df
    Actions:
        1. Removes outliers
            a. lower limit is Q1 - (1.5*IQR)
            b. upper limit is Q3 + (1.5*IQR)
        2. Drop nulls and duplicates
        3. Change column names
    Returns: cleaned df
    Modules:
        1. import scipy.stats as stats
        2. import pandas as pd
        3. import numpy as np
    '''
    
    # remove outliers
    # initialize dict
    outlier_limits = {}
    
    # for each column in df
    for col in df:
        
        if df[col].dtype != 'O':
            
            # set quartiles
            q1, q3 = df[col].quantile([.25, .75])

            # Set iqr 
            iqr = q3 - q1

            # add to dictionary with the upper limits and lower limits
            outlier_limits[col] =  {'low_limit_5': df[col].quantile(.05),
                                    'low_limit':  q1 - 1.5 * iqr,
                          'up_limit': q3 + 1.5 * iqr
                         }

    # for each cols
    for col in df:
        
        if col in outlier_limits:
            
            # remove all observations that exceed upper limit
            df = df[(df[col] <= outlier_limits[col]['up_limit'])]

            # remove all observations that are below the lower limit
            df = df[(df[col] >= outlier_limits[col]['low_limit'])]

    # drop nulls and duplicates
    df = df.dropna().drop_duplicates()
    
    # change fips codes to county names
    df.fips.replace(to_replace=[6037.0, 6059.0], value=['Los Angeles', 'Orange'], inplace=True)
    
    # change columns names
    df = df.rename(columns={'bedroomcnt': 'beds',
                  'bathroomcnt': 'baths',
                  'calculatedfinishedsquarefeet': 'square_feet',
                  'taxvaluedollarcnt': 'tax_value',
                            'fips': 'county'
                           })
    
    # adding option for insights from exploration
    if focus == True:
        
        # removes all half-bathrooms from the data
        df = df[df.baths.astype(str).str[-1] != '5']
    
    # exit function with clean df
    return df


def split_data(df):
    '''
    Arguments: clean dataframe
    Actions: splits Dataframe into a train, validate, and test datasets for explorations
    Returns: train, validate, and test datasets
    Modules:
        1. from sklearn.model_selection import train_test_split
    '''
    # splitting with test focus
    train_val, test = train_test_split(df, train_size=.8, random_state=1017)
    
    #splitting with train/validate focus
    train, validate = train_test_split(train_val, train_size=.7, random_state=1017)

    # exits function and returns train, validate, test
    return train, validate, test

def wrangle_zillow():
    '''
    Arguments: none
    Actions:
        1. Gets zillow data
        2. Cleans zillow data
        3. Splits zillow data
    Returns: train, validate, test
    Modules: get_zillow_data, clean_data, split_data
    '''
    # splits cleaned data into train, validate, test
    train, validate, test = split_data(
        
        # cleans data
        clean_data(
        
            # retrieves data
            get_zillow()))
    
    # exits function with wrangled data
    return train, validate, test

