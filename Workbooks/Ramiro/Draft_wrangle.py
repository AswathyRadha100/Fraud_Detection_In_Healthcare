# Import numpy for numerical operations
import pandas as pd
# Import Pandas for data manipulation
import numpy as np
# Import Matplotlib for data visualization
import matplotlib.pyplot as plt
# Import seaborn for data visualization
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder # did not use OneHotEncoder 


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

def DataFrame_shape(train, test, new_df):
    test_train_count = len(train) + len(test)
    print(f'Sum of both train and test -> {test_train_count}')
    print(f'Sum of new DataFrame -> {len(new_df)}')


# ======================================================================================

def beneficiary_label_encode(df):
	# make all columns lowercase 
    df.columns = df.columns.str.lower()

    # List of columns for label encoding 
    col_list = ['gender', 'race', 'renaldiseaseindicator',
				'chroniccond_alzheimer', 'chroniccond_heartfailure',
				'chroniccond_kidneydisease', 'chroniccond_cancer',
				'chroniccond_obstrpulmonary', 'chroniccond_depression',
				'chroniccond_diabetes', 'chroniccond_ischemicheart',
				'chroniccond_osteoporasis', 'chroniccond_rheumatoidarthritis',
				'chroniccond_stroke']

	# Label Encoding each column one by one

    for col in col_list:
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])
        

# ======================================================================================

def beneficiary_OneHotLable_encode(df): 
    df.columns = df.columns.str.lower()
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder_df = pd.DataFrame(encoder.fit_transform(df[['race']]).toarray())
    df = df.join(encoder_df)
    df = df.rename(columns={0: 'race_0'})
    df = df.rename(columns={1: 'race_1'})
    df = df.rename(columns={2: 'race_2'})
    df = df.rename(columns={3: 'race_3'})
    return df

# ======================================================================================

def prep_beneficiary_data(df):
    df.columns = df.columns.str.lower()
    df['dod'] = pd.to_datetime(df['dod'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['deceased'] = np.where((df['dod'].replace(np.nan, 'rep') == 'rep'), 0, 1)
    df['age'] = round(((df['dod'] - df['dob']).dt.days) / 365)
    df['age'] = df['age'].fillna(((pd.to_datetime('2009-12-01', format='%Y-%m-%d') - df['dob']).dt.days) / 365)
    df['age'] = df['age'].astype(int)
    df['total_reimbursed_amt'] = df['ipannualreimbursementamt'] + df['opannualreimbursementamt']
    df['total_deductible_amt'] = df['ipannualdeductibleamt'] + df['opannualreimbursementamt']
    df['dob_year'] = df['dob'].dt.year
    df['dob_month'] = df['dob'].dt.month
    df['dob_day'] = df['dob'].dt.day
    df = df.drop(['dob', 'dod'], axis=1)  # Assign the result back to the dataframe
    return df
