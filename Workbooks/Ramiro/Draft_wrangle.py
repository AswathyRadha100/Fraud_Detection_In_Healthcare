# Import numpy for numerical operations
import pandas as pd
# Import Pandas for data manipulation
import numpy as np
# Import Matplotlib for data visualization
import matplotlib.pyplot as plt
# Import seaborn for data visualization
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder 
from sklearn.model_selection import train_test_split  


def acquire_test_data():
    '''
    Load test datasets including outpatient, inpatient, beneficiary, and main test data.

    Returns:
    tuple: A tuple containing test_outpatient_df, test_inpatient_df, test_beneficiary_df, and test_df dataframes.
    '''
    test_outpatient_df = pd.read_csv('Test_Outpatientdata.csv')
    test_inpatient_df = pd.read_csv('Test_Inpatientdata.csv')
    test_beneficiary_df = pd.read_csv('Test_Beneficiarydata.csv')
    test_df = pd.read_csv('Test.csv')
    return test_outpatient_df, test_inpatient_df, test_beneficiary_df, test_df

# ======================================================================================

def acquire_train_data():
     """
    Load train datasets including outpatient, inpatient, beneficiary, and main train data.

    Returns:
    tuple: A tuple containing train_outpatient_df, train_inpatient_df, train_beneficiary_df, and train_df dataframes.
    """
    train_outpatient_df = pd.read_csv('Train_Outpatientdata.csv')
    train_inpatient_df = pd.read_csv('Train_Inpatientdata.csv')
    train_beneficiary_df = pd.read_csv('Train_Beneficiarydata.csv')
    train_df = pd.read_csv('Train.csv')
    return train_outpatient_df, train_inpatient_df, train_beneficiary_df, train_df

# ======================================================================================

def DataFrame_shape(train, test, new_df):
    """
    Calculate and print the number of rows in train, test, and a new dataframe.

    Args:
    train (DataFrame): The training dataframe.
    test (DataFrame): The test dataframe.
    new_df (DataFrame): The new dataframe to compare.

    Returns:
    None
    """
    test_train_count = len(train) + len(test)
    print(f'Sum of both train and test -> {test_train_count}')
    print(f'Sum of new DataFrame -> {len(new_df)}')


# ======================================================================================

def beneficiary_label_encode(df):
	"""
    Apply label encoding to selected columns in the beneficiary dataframe.

    Args:
    df (DataFrame): The beneficiary dataframe to be label encoded.

    Returns:
    DataFrame: The dataframe with label encoded columns.
    """
    # Make all columns lowercase
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

def beneficiary_OneHotLabel_encode(df): 
    """
    Apply one-hot encoding to the 'race' column in the beneficiary dataframe.

    Args:
    df (DataFrame): The beneficiary dataframe to be encoded.

    Returns:
    DataFrame: The dataframe with one-hot encoded 'race' column.
    """
    df.columns = df.columns.str.lower()
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder_df = pd.DataFrame(encoder.fit_transform(df[['race']]).toarray())
    df = df.join(encoder_df)
   
    columns_to_rename = {0: 'race_0', 1: 'race_1', 2: 'race_2', 3: 'race_3'}
    for old_col, new_col in columns_to_rename.items():
        df = df.rename(columns={old_col: new_col})
        df[new_col] = df[new_col].astype(int)
    return df

# ======================================================================================

def prep_beneficiary_data(df):
    """
    Preprocess the beneficiary data by performing the following steps:
    
    1. Convert column names to lowercase.
    2. Convert 'dod' and 'dob' columns to datetime objects.
    3. Create a 'deceased' column indicating whether the beneficiary is deceased (1) or not (0).
    4. Calculate the age of the beneficiary at a reference date.
    5. Calculate the total reimbursed amount as the sum of inpatient and outpatient annual reimbursement amounts.
    6. Calculate the total deductible amount as the sum of inpatient annual deductible amount and outpatient annual reimbursement amount.
    7. Create new features 'dob_year', 'dob_month', and 'dob_day' based on the 'dob' column.
    8. Drop the 'dob' and 'dod' columns.

    Args:
    df (DataFrame): The beneficiary dataframe to be preprocessed.

    Returns:
    DataFrame: The preprocessed beneficiary dataframe.
    """
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

# ======================================================================================

def wrangle_inpatient(df):
    """
    Preprocess the inpatient data by performing the following steps:
    
    1. Replace spaces in column names with underscores.
    2. Convert column names to lowercase.
    3. Impute null values in 'attendingphysician', 'operatingphysician', and 'otherphysician' with 'PHY000000'.
    4. Impute null values in 'clmdiagnosiscode_1' to 'clmdiagnosiscode_10' with '00000'.
    5. Impute null values in 'clmprocedurecode_1' to 'clmprocedurecode_3' with '000'.
    6. Drop columns 'clmprocedurecode_4', 'clmprocedurecode_5', 'clmprocedurecode_6' due to a high percentage of null values.
    7. Impute null values in 'deductibleamtpaid' with '1068'.

    Args:
    df (DataFrame): The inpatient dataframe to be preprocessed.

    Returns:
    DataFrame: The preprocessed inpatient dataframe.
    """
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.lower()

    # impute null values with PHY000000 for AttendingPhysician,OperatingPhysician,OtherPhysician 
    df['attendingphysician'] = df['attendingphysician'].fillna('PHY000000')
    df['operatingphysician'] = df['operatingphysician'].fillna('PHY000000')
    df['otherphysician'] = df['otherphysician'].fillna('PHY000000')
    
    # impute null values with '00000' for ClmDiagnosisCode_1 to ClmDiagnosisCode_10 in loop
    for i in range(1,11):
        df[f'clmdiagnosiscode_{i}'] = df[f'clmdiagnosiscode_{i}'].fillna('00000')

    
    # impute null values with '000' for ClmProcedureCode_1 to ClmProcedureCode_6 in loop
    for i in range(1,4):
        df[f'clmprocedurecode_{i}'] = df[f'clmprocedurecode_{i}'].fillna('000')    



    # drop columns ClmProcedureCode_4,ClmProcedureCode_5,ClmProcedureCode_6  as 99% of the values are null     
    df = df.drop(['clmprocedurecode_4','clmprocedurecode_5','clmprocedurecode_6'], axis=1)

    # impute null values with '1068' for DeductibleAmtPaid
    df['deductibleamtpaid'] = df['deductibleamtpaid'].fillna(1068)

    # rename columns clmprocedurecode_1,clmprocedurecode_2,clmprocedurecode_3 as clmprocedurecode_1,clmprocedurecode_2,clmprocedurecode_3 in the format clmprocedurecode_i_1 where i denotes inpatient
    df = df.rename(columns={'clmprocedurecode_1':'clmprocedurecode_1','clmprocedurecode_2':'clmprocedurecode_2','clmprocedurecode_3':'clmprocedurecode_3'})
  
    return df

# ======================================================================================


def wrangle_outpatient(df):
    """
    Preprocess outpatient data by performing the following steps:
    1. Replace spaces in column names with underscores.
    2. Convert column names to lowercase.
    3. Remove unnecessary columns ('clmdiagnosiscode_10', 'clmprocedurecode_1', ...).
    4. Impute null values with 'PHY000000' for 'AttendingPhysician', 'OperatingPhysician', and 'OtherPhysician'.
    5. Impute null values with '00000' for 'ClmDiagnosisCode_1' to 'ClmDiagnosisCode_9'.
    6. Impute null values with '00000' for 'ClmAdmitDiagnosisCode' (as 79% of the values are null).

    Args:
    df (DataFrame): The outpatient dataframe to be processed.

    Returns:
    DataFrame: The preprocessed outpatient dataframe.
    """
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.lower()

    df = df.drop(['clmdiagnosiscode_10','clmprocedurecode_1','clmprocedurecode_2','clmprocedurecode_3','clmprocedurecode_4','clmprocedurecode_5','clmprocedurecode_6'], axis=1)

    # impute null values with PHY000000 for AttendingPhysician,OperatingPhysician,OtherPhysician
    df['attendingphysician'] = df['attendingphysician'].fillna('PHY000000')
    df['operatingphysician'] = df['operatingphysician'].fillna('PHY000000')
    df['otherphysician'] = df['otherphysician'].fillna('PHY000000')

    # impute null values with '00000' from ClmDiagnosisCode_1 to ClmDiagnosisCode_9 in a  loop
    for i in range(1,10):
        df[f'clmdiagnosiscode_{i}'] = df[f'clmdiagnosiscode_{i}'].fillna('00000')


    # ClmAdmitDiagnosisCode impute it with '00000' as 79% of the values are null
    df['clmadmitdiagnosiscode'] = df['clmadmitdiagnosiscode'].fillna('00000')
   # rename columns clmprocedurecode_1 to clmprocedurecode_6 as  in the format clmprocedurecode_i_1 where i denotes inpatient
   # for i in range(1,7):
   #     df = df.rename(columns={f'clmprocedurecode_{i}':f'clmprocedurecode_i_{i}'})
    
    return df

# ======================================================================================

def wrangle_fraud(df):
    '''
    replace all empty space with an underscore
    make all the column names lowercase
    drop all NA for fraud
    encode the target variable 
    '''
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.lower()
    
    # Drop rows where 'potentialfraud' column has NaN values
    df = df.dropna(subset=['potentialfraud'])
    
    # Encode 'potentialfraud'
    df['potentialfraud_encoded'] = df['potentialfraud'].map({'Yes': 1, 'No': 0})
    return df

# ======================================================================================

def summarize_outliers(df, k=1.5) -> None:
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
    print('Descriptions:')
    # print the description of the df, transpose, output markdown
    #print(df.describe().T.to_markdown())
    print('======================\n======================')
    # lets do that for categorical info as well
    # we will use select_dtypes to look at just Objects
    print(df.select_dtypes('int64').describe().T.to_markdown())
    print('======================\n======================')
    print('missing values:')
    print('by column:')
    print(missing_by_col(df).to_markdown())
    print('by row: ')
    print(missing_by_row(df).to_markdown())
    print('======================\n======================')
    print('Outliers: ')
    print(report_outliers(df, k=k))
    print('======================\n======================')


def missing_by_col(df):
    """
    Count the number of missing values by column in a DataFrame.

    Args:
    df (DataFrame): The DataFrame to analyze.

    Returns:
    Series: A Series with column names as the index and the count of missing values as values.
    """
    return df.isnull().sum(axis=0)

def missing_by_row(df):
    """
    Generate a report on the number and percentage of rows with a certain number of missing columns.

    Args:
    df (DataFrame): The DataFrame to analyze.

    Returns:
    DataFrame: A DataFrame with columns 'num_cols_missing', 'percent_cols_missing', and 'num_rows'.
    """
    # get the number of missing elements by row (axis 1)
    count_missing = df.isnull().sum(axis=1)

    # get the ratio/percent of missing elements by row:
    percent_missing = round((df.isnull().sum(axis=1) / df.shape[1]) * 100)

    # make a df with those two series (same len as the original df)
    # reset the index because we want to count both things
    # under aggregation (because they will always be sononomous)
    # use a count function to grab the similar rows
    # print that dataframe as a report
    rows_df = pd.DataFrame({
    'num_cols_missing': count_missing,
    'percent_cols_missing': percent_missing
    }).reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).\
    count().reset_index().rename(columns={'index':'num_rows'})
    return rows_df

def get_fences(df, col, k=1.5) -> tuple:
    """
    Calculate upper and lower fences for identifying outliers in a numeric column.

    Args:
    df (DataFrame): The DataFrame containing the column.
    col (str): The name of the column to analyze.
    k (float): The threshold multiplier for the IQR.

    Returns:
    float: Lower bound for outliers.
    float: Upper bound for outliers.
    """
    q3 = df[col].quantile(0.75)
    q1 = df[col].quantile(0.25)
    iqr = q3 - q1
    upper_bound = q3 + (k * iqr)
    lower_bound = q1 - (k * iqr)
    return lower_bound, upper_bound

def report_outliers(df, k=1.5) -> None:
    """
    Report outliers in numeric columns of a DataFrame based on the specified threshold.

    Args:
    df (DataFrame): The DataFrame to analyze.
    k (float): The threshold for identifying outliers.

    Returns:
    None
    """
    num_df = df.select_dtypes('number')
    for col in num_df:
        if len(num_df[col].value_counts()) > 20:
            lower_bound, upper_bound = get_fences(df,col, k=k)
            print(f'Outliers for Col {col}:')
            print('lower: ', lower_bound, 'upper: ', upper_bound)
            print(df[col][(
                df[col] > upper_bound) | (df[col] < lower_bound)])
            print('----------')
            
def get_continuous_feats(df) -> list:
    """
    Find continuous numerical features in a DataFrame.

    Args:
    df (DataFrame): The DataFrame to analyze.

    Returns:
    list: List of column names containing continuous numerical features.
    """
    num_cols = []
    num_df = df.select_dtypes('number')
    for col in num_df:
        if num_df[col].nunique() > 20:
            num_cols.append(col)
    return num_cols

# ======================================================================================

def split_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split a DataFrame into training, validation, and test sets with optional stratification.

    Args:
    df (DataFrame): The DataFrame to split.
    target (Series): Optional target variable for stratified splitting.

    Returns:
    train (DataFrame): Training data.
    validate (DataFrame): Validation data.
    test (DataFrame): Test data.
    """
    train, test = train_test_split(df, test_size=.15, random_state=117, stratify=df.potentialfraud)
    train, validate = train_test_split(train, test_size=.15, random_state=117, stratify=train.potentialfraud)
    return train, validate, test

# ======================================================================================

def display_numeric_column_histograms(data_frame):
    """
    Display histograms for numeric columns in a DataFrame with three colors.

    Args:
    data_frame (DataFrame): The DataFrame to visualize.

    Returns:
    None(prints to console)
    """
    numeric_columns = data_frame.select_dtypes(exclude=["object", "category"]).columns.to_list()
    # Define any number of colors for the histogram bars
    colors = ["#FFBF00"]
    for i, column in enumerate(numeric_columns):
        # Create a histogram for each numeric column with two colors
        figure, axis = plt.subplots(figsize=(10, 3))
        sns.histplot(data_frame, x=column, ax=axis, color=colors[i % len(colors)])
        axis.set_title(f"Histogram of {column}")
        plt.show()

# ======================================================================================


# function to create a new feature for inpatient dataframes
def create_features_inpatient(df):
    """
    Create new features for inpatient dataframes, including the following:
    
    1. Convert date columns to datetime objects: 'inpatient_claimstartdt' and 'claimenddt'.
    2. Calculate the claim duration in days.
    3. Create a new feature 'NumPhysicians' by counting non-null values in physician-related columns.
    4. Change the dtype of date columns: 'inpatient_claimstartdt', 'claimenddt', 'admissiondt', 'dischargedt' to 'datetime64'.

    Args:
    df (DataFrame): The inpatient dataframe to which features will be added.

    Returns:
    DataFrame: The inpatient dataframe with the new features.
    """
    # Convert the date columns to datetime objects
    df['claimstartdt'] = pd.to_datetime(df['claimstartdt'])
    df['claimenddt'] = pd.to_datetime(df['claimenddt'])

    # Calculate the Claim Duration
    df['claimduration'] = (df['claimenddt'] - df['claimstartdt']).dt.days
    
    # Create a new feature "NumPhysicians" by counting non-null values in the relevant columns
    df['numphysicians'] = df[['attendingphysician', 'operatingphysician', 'otherphysician']].count(axis=1)

    # change the dtype of "claimstartdt" ,"claimenddt","admissiondt","dischargedt" to "datetime64"
    df['claimstartdt'] = pd.to_datetime(df['claimstartdt'])
    df['claimenddt'] = pd.to_datetime(df['claimenddt'])
    df['admissiondt'] = pd.to_datetime(df['admissiondt'])
    df['dischargedt'] = pd.to_datetime(df['dischargedt'])
    return df

# ======================================================================================

# function to create a new feature for outpatient dataframes
def create_features_outpatient(df):
    """
    Create new features for outpatient dataframes, including the following:
    
    1. Convert date columns to datetime objects: 'outpatient_claimstartdt' and 'claimenddt'.
    2. Calculate the claim duration in days.

    Args:
    df (DataFrame): The outpatient dataframe to which features will be added.

    Returns:
    DataFrame: The outpatient dataframe with the new features.
    """
    # Convert the date columns to datetime objects
    df['claimstartdt'] = pd.to_datetime(df['claimstartdt'])
    df['claimenddt'] = pd.to_datetime(df['claimenddt'])
    # Calculate the Claim Duration
    df['claimduration'] = (df['claimenddt'] - df['claimstartdt']).dt.days
    return df

# ======================================================================================

# function to create a new feature "ChronicDiseaseCount" from the "ChronicCond" features for beneficiary dataframe
def create_chronic_disease_count_feature_beneficiary(df):
    """
    Create a new feature "ChronicDiseaseCount" for beneficiary dataframes by counting the number of "1" values (which means "yes") in the relevant columns.

    Args:
    df (DataFrame): The beneficiary dataframe to which the feature will be added.

    Returns:
    DataFrame: The beneficiary dataframe with the new "ChronicDiseaseCount" feature.
    """
    # Create a new feature "ChronicDiseaseCount" by counting the number of "1" values(which means "yes") in the relevant columns
    df['chronicdiseasecount'] = df[['chroniccond_alzheimer', 'chroniccond_heartfailure', 'chroniccond_kidneydisease', 'chroniccond_cancer', 'chroniccond_obstrpulmonary', 'chroniccond_depression', 'chroniccond_diabetes', 'chroniccond_ischemicheart', 'chroniccond_osteoporasis', 'chroniccond_rheumatoidarthritis', 'chroniccond_stroke']].apply(lambda row: row.str.contains('1').sum(), axis=1)
    return df

# ======================================================================================

def merge_inpatient_fraud(beneficiary, inpatient, fraud):
    '''
    join beneficiary, inpatient and fraud datasets together to 
    create inpatent dataframe for exploration. 
    '''
    df = beneficiary.join(inpatient.set_index('beneid'), on='beneid', how='left')
    df = df.join(fraud.set_index('provider'), on='provider', how='left')
    df = df.dropna(subset=['potentialfraud'])
    df.reset_index(inplace = True, drop = True)
    df['potentialfraud_encoded'] =  df['potentialfraud_encoded'].astype(int)
    return df

# ======================================================================================

def merge_outpatient_fraud(beneficiary, outpatient, fraud):
    '''
    join beneficiary, outpatient and fraud datasets together to 
    create outpatent dataframe for exploration. 
    '''
    df = beneficiary.join(outpatient.set_index('beneid'), on='beneid', how='left')
    df = df.join(fraud.set_index('provider'), on='provider', how='left')
    df = df.dropna(subset=['potentialfraud'])
    df.reset_index(inplace = True, drop = True)
    df['potentialfraud_encoded'] =  df['potentialfraud_encoded'].astype(int)
    return df 