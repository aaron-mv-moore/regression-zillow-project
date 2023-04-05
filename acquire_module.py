# this module contains multiple functions for various data acquisition needs
# the information must be replaced with the correct sgtrings when becessary

def get_data_sql():
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
    Modules: pandas, os, env
    '''
    # a variable to hold the xpected or future file name
    filename = '''ADD FILE NAME'''
    
    # if the file is present in the directory 
    if os.path.isfile(filename):
      
        # read the csv and assign it to the variable df
        df = pd.read_csv(filename)
        
        # return the dataframe and exit the funtion
        return df
    
    # if the file is not in the current working directory,
    else:
        # assign the name of the database to db
        db = '''ADD DB NAME'''
        
        # use the env.py function to get the url needed from the db
        url = env.get_db_url(db)
        
        # assign the sql query into the variable query
        query = '''ADD QUERY'''
        
        # query sql using pandas function
        df = pd.read_sql(query, url)
        
        # save the dataframe as a csv to the current working directory
        df.to_csv(filename)
        
        # returns the dataframe
        return df


def get_data_csv():
    '''
    Argument: No arguments required
    Actions: 
        1. Checks for the existence of the csv in the current directory
            a. if present:
                i. reads the csv from the current working directory
            b. if not present:
                i. reads csv from url
                ii. saves the csv to the current working directory
    Return: dataframe
    Modules: pandas, os
    '''
    # a variable to hold the xpected or future file name
    filename = 'FILENAME.CSV'
    
    # if the file is present in the directory 
    if os.path.isfile(filename):
      
        # read the csv and assign it to the variable df
        df = pd.read_csv(filename)
        
        # return the dataframe and exit the funtion
        return df
    
    # if the file is not in the current working directory,
    else:
        
        # url needed to read from a csv
        url = 'ADDURL'
        
        # reads crz from url using pandas function
        df = pd.read_csv(url)
        
        # save the dataframe as a csv to the current working directory
        df.to_csv(filename)
        
        # returns the dataframe
        return df
