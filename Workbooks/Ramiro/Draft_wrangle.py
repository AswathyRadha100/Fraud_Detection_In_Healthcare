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
    return test_outpatient_df, test_inpatient_df, test_beneficiary_df, test_df

def acquire_train_data():
    train_outpatient_df = pd.read_csv('Train_Outpatientdata.csv')
    train_inpatient_df = pd.read_csv('Train_Inpatientdata.csv')
    train_beneficiary_df = pd.read_csv('Train_Beneficiarydata.csv')
    train_df = pd.read_csv('Train.csv')
    return train_outpatient_df, train_inpatient_df, train_beneficiary_df, train_df

