# Import numpy for numerical operations
import pandas as pd
# Import Pandas for data manipulation
import numpy as np
# Import Matplotlib for data visualization
import matplotlib.pyplot as plt
# Import seaborn for data visualization
import seaborn as sns

def acquire_test_data():
    test_outpatient_df = pd.read_csv('Test_Outpatientdata-1542969243754.csv')
    test_inpatient_df = pd.read_csv('Test_Inpatientdata-1542969243754.csv')
    test_beneficiary_df = pd.read_csv('Test_Beneficiarydata-1542969243754.csv')
    test_df = pd.read_csv('Test-1542969243754.csv')
    return test_outpatient_df, test_beneficiary_df, test_inpatient_df, test_df

def acquire_train_data():
    train_outpatient_df = pd.read_csv('Train_Outpatientdata-1542865627584.csv')
    train_inpatient_df = pd.read_csv('Train_Inpatientdata-1542865627584.csv')
    train_benificiart_df = pd.read_csv('Train_Beneficiarydata-1542865627584.csv')
    train_df = pd.read_csv('Train-1542865627584.csv')
    return train_outpatient_df, train_inpatient_df, train_benificiart_df, train_df