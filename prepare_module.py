import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def split_data(df, stratify_on=None):
    '''
    Arguments: prepared dataframe, optional target - must be a string literal that is a column title
    Actions: 
        1. Splits the dataframe with 80% of the data assigned to tv and 20% assigned to test
        2. Splits the tv dataset with 70% of tv assigned to train and 30% assigned to validate
    Returns: 3 variables, each containing a portion
    Modules: from sklearn.model_selection import train_test_split, pandas as pd
    Note: Order matters with variable assignment
    '''
    
    # when the target is a string that is a column title
    if stratify_on in df.columns.to_list():
        # the data is split 80/20 with the target used for stratification
        train_validate, test = train_test_split(df, train_size=.8, random_state = 1017,
                stratify = df[stratify_on])
        
         # splitting train_validate 70/30 with the target used for stratification
        train, validate = train_test_split(train_validate, train_size=.7, stratify=train_validate[stratify_on])
    # for all other targets
    else:
        # inform user that there is no stratification
        print('No stratification applied during the split')
        
        # split that data 80/20
        train_validate, test = train_test_split(df, train_size=.8, random_state = 1017)
        
        # splitting train_validate 70/30
        train, validate = train_test_split(train_validate, train_size=.7)
    
    return train, validate, test


def prepare_split(df, base_explore=True):
    '''
    Arguments: cleaned df, base_explore retains variables in a non-encoded format, useful for visualizations and exploration
    Actions:
        1. Creates a dataframe with only dummy variables, numerical variables, and the target
        2. Formats all the column titles for python usability
        3. Splits data into train validate, and test with straitification on target
    Return: train, validate, test
    Modules: pandas as pd
    '''
    # assigning a target
    target = 'ADD TARGET'
    
    # default argument fo base_explore is True
    if base_explore == True:

        # skip the encoding of the variables
        pass
    
    else:
        # Create list of object type/categorical columns
        df_objects = [col for col in df if df[col].dtype == 'O' and col != target]
        
        # Create dummy variables and add them to the df
        df = pd.concat([df, pd.get_dummies(df[df_objects], drop_first=True)], axis=1)
    
        # Create a list of all non-object variables and including the target
        num_cols = [col for col in df if df[col].dtype != 'O' or col == target]

        # creating a df with only the variables needed for exploring and modeling
        df = df[num_cols]

        # change the titles of the encoded variables to be python friendly
        df.columns = df.columns.str.lower().str.strip().str.replace(' ','_')
    
    # edit target series to reflect 1,0 for values
    df[target].replace(['ADD VALUE', 'ADD VALUE'], [1, 0], inplace=True)
    
    # splits the df into train, validate, and test with a stratification on the target
    train, validate, test = split_data(df, stratify_on=target)
    
    return train, validate, test
