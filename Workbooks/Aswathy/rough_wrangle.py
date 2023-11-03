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

    # impute null values with '00000' from ClmDiagnosisCode_1 to ClmDiagnosisCode_9 in a  loop
    for i in range(1,10):
        df[f'clmdiagnosiscode_{i}'] = df[f'clmdiagnosiscode_{i}'].fillna('00000')


    # ClmAdmitDiagnosisCode impute it with '00000' as 79% of the values are null
    df['clmadmitdiagnosiscode'] = df['clmadmitdiagnosiscode'].fillna('00000')
    return df
#____________________________________________________________________
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
    # _____________________________________________________________



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
def split_data(df, target=None) -> tuple:
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
    train_val, test = train_test_split(
        df,
        train_size=0.8,
        random_state=1349,
        stratify=target)
    train, validate = train_test_split(
        train_val,
        train_size=0.7,
        random_state=1349,
        stratify=target)
    return train, validate, test
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



# function to create a new feature for inpatient dataframes
def create_features_inpatient(df):
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


def create_features_outpatient(df):
    # Convert the date columns to datetime objects
    df['claimstartdt'] = pd.to_datetime(df['claimstartdt'])
    df['claimenddt'] = pd.to_datetime(df['claimenddt'])
    # Calculate the Claim Duration
    df['claimduration'] = (df['claimenddt'] - df['claimstartdt']).dt.days
   
    return df



# function to create a new feature "ChronicDiseaseCount" from the "ChronicCond" features for beneficiary dataframe
def create_chronic_disease_count_feature_beneficiary(df):
    # Create a new feature "ChronicDiseaseCount" by counting the number of "1" values(which means "yes") in the relevant columns
    df['chronicdiseasecount'] = df[['chroniccond_alzheimer', 'chroniccond_heartfailure', 'chroniccond_kidneydisease', 'chroniccond_cancer', 'chroniccond_obstrpulmonary', 'chroniccond_depression', 'chroniccond_diabetes', 'chroniccond_ischemicheart', 'chroniccond_osteoporasis', 'chroniccond_rheumatoidarthritis', 'chroniccond_stroke']].apply(lambda row: row.str.contains('1').sum(), axis=1)
    return df

