# get data
from preprocess import preprocess_zillow

# tab data
import pandas as pd
import numpy as np

# stats and modeling needs
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

import warnings
warnings.filterwarnings("ignore")


# getting scaled data
X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test = preprocess_zillow()

def get_baseline():
    '''
    Actions: caluculate baseline and add it to train df, output the rmse and rmse difference
    '''
    y='tax_value'
    # baseline using mean
    y_train['baseline_mean'] = round(y_train[y].mean(), 2)
    y_validate['baseline_mean'] = round(y_validate.loc[:,y].mean(), 2)
    
    # RMSE of mean baseline predictions
    rmse_train = mean_squared_error(y_train.loc[:,y], y_train['baseline_mean'], squared=False)
    rmse_validate = mean_squared_error(y_validate.loc[:,y], y_validate['baseline_mean'], squared=False)

    # getting difference
    rmse_diff = rmse_train - rmse_validate
    
    # printing scores
    print(f'''Baseline Model
    Baseline: {round(y_train[y].mean(), 2)}
    RMSE on Train: {round(rmse_train, 2)}
    RMSE on Validate: {round(rmse_validate, 2)}
    RMSE Difference: {round(rmse_diff, 2)}''')
    
    return

def basic_linear_modeling(X_train_scaled, y_train, x_validate_scaled, y_validate, model):
    '''
     Actions: runs through specified linear model pipeline and returns the predictions for train and val
    '''
    # fit the model
    model.fit(X_train_scaled, y_train)
   
    # return predictions
    return model.predict(X_train_scaled), model.predict(X_validate_scaled)

def get_lassolars_model():
    '''
    Actions: Returns rmse metric for lassolars model
    '''
    
    # assigning model to y_train
    y_train['lars_alpha1_preds'], y_validate['lars_alpha1_preds'] = basic_linear_modeling(X_train_scaled, y_train[['tax_value']], X_validate_scaled, y_validate[['tax_value']], model=LassoLars(alpha=1))
    
    # RMSE of mean predictions
    rmse_train = mean_squared_error(y_train['tax_value'], y_train['lars_alpha1_preds'], squared=False)
    rmse_validate = mean_squared_error(y_validate['tax_value'], y_validate['lars_alpha1_preds'], squared=False)

    # getting difference
    rmse_diff = rmse_train - rmse_validate
    
    # printing scores
    print(f'''LassoLars Alpha-1 Model
    RMSE on Train: {round(rmse_train, 4)}
    RMSE on Validate: {round(rmse_validate, 4)}
    RMSE Difference: {round(rmse_diff, 4)}''')
    
    return


def get_poly_model():
    '''
    Action: Returns metrics for polynomial model
    '''
    
    # make the polyni mial features
    pf = PolynomialFeatures(degree=2)

    # fit_transform
    X_train_degree2 = pf.fit_transform(X_train_scaled)

    # transform for validate
    X_validate_degree2 = pf.transform(X_validate_scaled)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.tax_value)

    # predict train
    y_train['poly_preds'] = lm2.predict(X_train_degree2)

    # predict validate
    y_validate['poly_preds'] = lm2.predict(X_validate_degree2)

    # RMSE of predictions
    rmse_train = mean_squared_error(y_train['tax_value'], y_train['poly_preds'], squared=False)
    rmse_validate = mean_squared_error(y_validate['tax_value'], y_validate['poly_preds'], squared=False)

    # getting difference
    rmse_diff = rmse_train - rmse_validate
    
    # printing scores
    print(f'''Polynomial 2-degree Model
    RMSE on Train: {round(rmse_train, 4)}
    RMSE on Validate: {round(rmse_validate, 4)}
    RMSE Difference: {round(rmse_diff, 4)}''')
    
    return


def get_ols_model():
    '''
    Action: prints metrics for ols model for view
    '''
    # assigns ols predictions to y datasets
    y_train['ols_preds'], y_validate['ols_preds'] = basic_linear_modeling(X_train_scaled, y_train[['tax_value']], X_validate_scaled, y_validate[['tax_value']], model=LinearRegression())
    
    # RMSE of ols model predictions
    rmse_train = mean_squared_error(y_train['tax_value'], y_train['ols_preds'], squared=False)
    rmse_validate = mean_squared_error(y_validate['tax_value'], y_validate['ols_preds'], squared=False)

    # getting difference
    rmse_diff = rmse_train - rmse_validate
    
    # printing scores
    print(f'''OLS Model
    RMSE on Train: {round(rmse_train, 4)}
    RMSE on Validate: {round(rmse_validate, 4)}
    RMSE Difference: {round(rmse_diff, 4)}''')
    
    return

def get_poly_test():
    '''
    Action: Returns metrics for polynomial model ran on test
    '''
    
    # make the polyni mial features
    pf = PolynomialFeatures(degree=2)

    # fit_transform
    X_train_degree2 = pf.fit_transform(X_train_scaled)

    # transform for validate
    X_test_degree2 = pf.transform(X_test_scaled)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.tax_value)

    # predict train
    y_train['poly_preds'] = lm2.predict(X_train_degree2)

    # predict validate
    y_test['poly_preds'] = lm2.predict(X_test_degree2)

    # RMSE of predictions
    rmse_train = mean_squared_error(y_train['tax_value'], y_train['poly_preds'], squared=False)
    rmse_test = mean_squared_error(y_test['tax_value'], y_test['poly_preds'], squared=False)

    # getting difference
    rmse_diff = rmse_train - rmse_test
    
    # printing scores
    print(f'''Polynomial 2-degree Model
    RMSE on Train: {round(rmse_train, 4)}
    RMSE on Test: {round(rmse_test, 4)}
    RMSE Difference: {round(rmse_diff, 4)}''')
    
    return
