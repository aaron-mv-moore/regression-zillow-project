import env
import pandas as pd
import os

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
        df = pd.read_csv(filename)

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
                      taxvaluedollarcnt
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