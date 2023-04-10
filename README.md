# Zillow Project: Linear Regression

## Description
- This project will us linear regression models to predict the tax value for properties using a Zillow dataset. It will go through the data science pipeline and create a model that accurately predicts the tax value of properties in the dataset.

## Goals
- Predict the values of single unit properties using the obervations from 2017
  - Construct a ML linear regression model that accurately predicts values
  - Deliver a report that a data scientist can read through and understand 

## Initial Questions
1. Can we predict the tax values of single unit properties using the obervations from 2017?
2. What features are drivers for the values of single unit properties?

## Plan
- Acquire data from MySQL
- Prepare data
  - Remove unnecessary features
  - Identify and replace missing values
  - Alter innapropriate data types
- Explore the data to find drivers and answer intital questions
- Create a model to predict values
  - Use features identified in explore to build predictive models
  - Evaluate models on train and validate data
  - Select the best model based on drivers identified in exploration
  - Evaluate the best model on test data
- Conclude with recommendations and next steps

## Data Dictionary
| Feature | Definition | 
|:--------|:-----------|
| beds | The number of bedrooms |
| baths | The number of bathrooms |
| square_feet | The number of calculated finished square feet |
| tax_value | Total tax assessed value of the property |
| county | County name based on the FIPS (Federal Information Processing Standards) code |


## Steps to Reproduce
1. Clone this repo
2. Insert credentials in the blank_eny.py file
3. Save blank_env.py as env.py
4. Run notebook

## Takeaways
- Our tax value is not normally ditributed
- There is a difference in the tax value of properties in Orange County and Los Angeles County
- There is a stonger correlation between square feet and tax value for homes in Orange County than Los Angeles County proerties
- Drivers discovered did lead to the success of performing better than the baseline for properties in Orange County

## Recommendations
- Find data that represents smaller geographic areas for each property, such as zip codes, due to the wide range of tax values in a single county
- Extend the life of this project to increase time for feature selection and featuren engineering
