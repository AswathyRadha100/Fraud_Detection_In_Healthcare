# Import numpy for numerical operations
import pandas as pd
# Import Pandas for data manipulation
import numpy as np
# Import Matplotlib for data visualization
import matplotlib.pyplot as plt
# Import seaborn for data visualization
import seaborn as sns

def acquire_test_data():
    test_outpatient_df = pd.read_csv('Test_Outpatientdata.csv')
    test_inpatient_df = pd.read_csv('Test_Inpatientdata.csv')
    test_beneficiary_df = pd.read_csv('Test_Beneficiarydata.csv')
    test_df = pd.read_csv('Test.csv')
    return test_outpatient_df, test_beneficiary_df, test_inpatient_df, test_df

def acquire_train_data():
    train_outpatient_df = pd.read_csv('Train_Outpatientdata.csv')
    train_inpatient_df = pd.read_csv('Train_Inpatientdata.csv')
    train_beneficiary_df = pd.read_csv('Train_Beneficiarydata.csv')
    train_df = pd.read_csv('Train.csv')
    return train_outpatient_df, train_inpatient_df, train_beneficiary_df, train_df

def summarize(df) -> None:
    '''
    Summarize will take in a pandas DataFrame
    and print summary statistics:
    info
    shape
    outliers
    description
    missing data stats
    
    Args:
    df (DataFrame): The DataFrame to be summarized.
    k (float): The threshold for identifying outliers.
    
    return: None (prints to console)
    '''
    # print info on the df
    print('Shape of Data: ')
    print(df.shape)
    print('======================\n======================')
    print('Info: ')
    print(df.info())
    print('======================\n======================')
    
    
    # Calculate missing values and percentages
    missing_values = df.isnull()
    missing_count = missing_values.sum()
    missing_percentage = (missing_count / len(df)) * 100
    
    print('Missing Data Stats:')
    print('Missing Data Count by Column:')
    print(missing_count)
    print('Missing Data Percentage by Column:')
    print(missing_percentage)

