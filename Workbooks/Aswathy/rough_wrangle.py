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

def DataFrame_shape(train, test, new_df):
    test_train_count = len(train) + len(test)
    print(f'Sum of both train and test -> {test_train_count}')
    print(f'Sum of new DataFrame -> {len(new_df)}')




def wrangle_inpatient(df):
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.lower()

    # impute null values with PHY000000 for AttendingPhysician,OperatingPhysician,OtherPhysician 
    df['attendingphysician'] = df['attendingphysician'].fillna('PHY000000')
    df['operatingphysician'] = df['operatingphysician'].fillna('PHY000000')
    df['otherphysician'] = df['otherphysician'].fillna('PHY000000')

    # impute null values with '00000' for ClmDiagnosisCode_1 to ClmDiagnosisCode_10
    df['clmdiagnosiscode_1'] = df['clmdiagnosiscode_1'].fillna('00000')
    df['clmdiagnosiscode_2'] = df['clmdiagnosiscode_2'].fillna('00000')
    df['clmdiagnosiscode_3'] = df['clmdiagnosiscode_3'].fillna('00000')
    df['clmdiagnosiscode_4'] = df['clmdiagnosiscode_4'].fillna('00000')
    df['clmdiagnosiscode_5'] = df['clmdiagnosiscode_5'].fillna('00000')
    df['clmdiagnosiscode_6'] = df['clmdiagnosiscode_6'].fillna('00000')    
    df['clmdiagnosiscode_7'] = df['clmdiagnosiscode_7'].fillna('00000')
    df['clmdiagnosiscode_8'] = df['clmdiagnosiscode_8'].fillna('00000')
    df['clmdiagnosiscode_9'] = df['clmdiagnosiscode_9'].fillna('00000')
    df['clmdiagnosiscode_10'] = df['clmdiagnosiscode_10'].fillna('00000')

    # impute null values with '000' for ClmProcedureCode_1 to ClmProcedureCode_6
    df['clmprocedurecode_1'] = df['clmprocedurecode_1'].fillna('000')
    df['clmprocedurecode_2'] = df['clmprocedurecode_2'].fillna('000')
    df['clmprocedurecode_3'] = df['clmprocedurecode_3'].fillna('000')

    # drop columns ClmProcedureCode_4,ClmProcedureCode_5,ClmProcedureCode_6  as 99% of the values are null     
    df = df.drop(['clmprocedurecode_4','clmprocedurecode_5','clmprocedurecode_6'], axis=1)

    # impute null values with '1068' for DeductibleAmtPaid
    df['deductibleamtpaid'] = df['deductibleamtpaid'].fillna(1068)
    return df



def wrangle_outpatient(df):
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.lower()

    df = df.drop(['clmdiagnosiscode_10','clmprocedurecode_1','clmprocedurecode_2','clmprocedurecode_3','clmprocedurecode_4','clmprocedurecode_5','clmprocedurecode_6'], axis=1)

    # impute null values with PHY000000 for AttendingPhysician,OperatingPhysician,OtherPhysician
    df['attendingphysician'] = df['attendingphysician'].fillna('PHY000000')
    df['operatingphysician'] = df['operatingphysician'].fillna('PHY000000')
    df['otherphysician'] = df['otherphysician'].fillna('PHY000000')

    # impute null values with '00000' for ClmDiagnosisCode_1 to ClmDiagnosisCode_9
    df['clmdiagnosiscode_1'] = df['clmdiagnosiscode_1'].fillna('00000')
    df['clmdiagnosiscode_2'] = df['clmdiagnosiscode_2'].fillna('00000')
    df['clmdiagnosiscode_3'] = df['clmdiagnosiscode_3'].fillna('00000')
    df['clmdiagnosiscode_4'] = df['clmdiagnosiscode_4'].fillna('00000')
    df['clmdiagnosiscode_5'] = df['clmdiagnosiscode_5'].fillna('00000')
    df['clmdiagnosiscode_6'] = df['clmdiagnosiscode_6'].fillna('00000')
    df['clmdiagnosiscode_7'] = df['clmdiagnosiscode_7'].fillna('00000')
    df['clmdiagnosiscode_8'] = df['clmdiagnosiscode_8'].fillna('00000')
    df['clmdiagnosiscode_9'] = df['clmdiagnosiscode_9'].fillna('00000')

    # ClmAdmitDiagnosisCode impute it with '00000' as 79% of the values are null
    df['clmadmitdiagnosiscode'] = df['clmadmitdiagnosiscode'].fillna('00000')
    return df
