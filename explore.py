# imports
from wrangle import wrangle_zillow
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stats

# getting data
train, validate, test = wrangle_zillow()

# creating masks for plotting
la_mask = (train.county == 'Los Angeles')
oc_mask = (train.county == 'Orange')
target = 'tax_value'

# What does the distribution of proerty tax value amount look like in our train data?
def get_tax_hist(): 
    '''
    Actions: gets a histogram of the tax value in the train dataset
    '''
    # initialize target variable
    target = 'tax_value'

    # getting histogram of tax value in train
    sns.histplot(data=train,
               x=target)
    plt.title('Tax Value Distribution')
    plt.show()
    
    return


# Is there a difference in the the tax value amount for homes in different counties?

def get_county_box():
    '''
    Actions: gets box plot with the counties tax values
    '''
    
    # get a box plot of train data with xx as the counties
    sns.boxplot(data=train,
                x='county',
               y=target)
    plt.title('Tax Value by County')
    
    # a line to represent the mean
    plt.axhline(train[target].mean(), c='r')
    plt.show()
    
    return

# making dict of train ds by county
county_data = {
    'LA': train[la_mask],
    'OC': train[oc_mask]
}

def get_county_stat():
    '''
    Actions: Shows the statisticto support the difference between LA and OC
    '''
    # run the stats test
    mw, p = stats.mannwhitneyu(county_data['LA'][target], county_data['OC'][target])
    
    # print results
    print(f'''
    Mann-Whitney: {mw}
    p-value: {p}
    ''')
    
    return


# 3) What is the relationship between tax value and square feet in Orange County? Los Angeles County?
def get_tax_sqft_scatter():
    '''
    Action: Show scatter plot of property tax values and square feet
    '''
    
    # set col variable
    col = 'square_feet'
    
    # create a plot with regression line and scatter
    sns.lmplot(data=train[la_mask].sample(1000),
               x=col,
               y=target, line_kws={'color': 'orange'})
    
    # set the mean as a horizontal red line
    plt.axhline(train[target].mean(), c='r')
    plt.title('Relationship between \nTax Value & Square Feet in LA County')
    plt.show()
    
    # create a scatter plot with regression line
    sns.lmplot(data=train[oc_mask].sample(1000),
               x=col,
               y=target, line_kws={'color': 'orange'})
    
    # set the mean as a horizontal red line
    plt.axhline(train[target].mean(), c='r')
    plt.title('Relationship between \nTax Value & Square Feet in Orange County')
    plt.show()
    
    return

#3) What is the relationship between tax value and square feet in Orange County? Los Angeles County?
def get_tax_sqft_stat():
    '''
    Actions: Shows the statisticts support thethe relationship between tax value and square feet in LA county
    '''
    # run the stats test
    rl, pl = stats.spearmanr(county_data['LA']['square_feet'], county_data['LA'][target])
    
    ro, po = stats.spearmanr(county_data['OC']['square_feet'], county_data['OC'][target])
    
    # print results
    print(f'''Los Angeles County
    Spearman r: {rl}
    p-value: {pl}
    
Orange County
    Spearman r: {ro}
    p-value: {po}
    ''')
    
    return
