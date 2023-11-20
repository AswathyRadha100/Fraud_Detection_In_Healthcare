--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                                                                                                                      #
# Healthcare Fraud Buster                                            
#                                                                                                                                                                      # 
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Project Description** 

###  The central aim of this project is to enhance the affordability of healthcare for millions of Americans by addressing healthcare fraud in the Medicare system. The project's core objective is to identify, prevent, and mitigate fraudulent practices within Medicare, leading to a decrease in the financial strain on the government, as Medicare is a government-funded healthcare program designed to provide health insurance to eligible individuals, older people or people with disability.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Project Goal** 

###  The main goals of the project are
  - &#9733; Develop a ML model for healthcare fraud detection
  - &#9733; Identify key drivers of healthcare fraud
  - &#9733; Improve fraud detection accuracy
  - &#9733; Provide insights into healthcare fraud variations
  - &#9733; Deliver a comprehensive report to the data science team responsible for addressing healthcare fraud  


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **Initial Thoughts**

###    In the initial phases of the project, data collection and preprocessing are paramount, ensuring the availability of high-quality healthcare data for analysis. This foundational stage lays the groundwork for harnessing data effectively in predicting healthcare fraud.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Skill Set Used **

- &#9733; Skill Set used
     - &#9642; Python
     - &#9642; NumPy
     - &#9642; Seaborn
     - &#9642; Scikit Learn
     - &#9642; Matplotlib

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## **Deliverables **

- &#9733; List of deliverables:
     - &#9642; Final Report
     - &#9642; Slide Show

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## **Modeling **

- &#9733; List of models:
     - &#9642; Decision Tree
     - &#9642; K Nearest Neighbours
     - &#9642; Linear Regression
     - &#9642; Logistic Regression
- &#9733; Best model for Inpatient dataset:
     - &#9642; Best Model is KNearest Neighbor
- &#9733; Best model for Outpatient dataset:
     - &#9642; Best Model Logistic Regression
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **The Plan**
- &#9733; Acquire data from https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis?select=Train-1542865627584.csv
- &#9733; Prepare data that's acquired:
  -  Feature Selection/Engineering
     - &#9642; Replace spaces with underscores
     - &#9642; Created new binary feature for fraud or legitimate claim 
     - &#9642; Created new feature that shows Patient type 
     - &#9642; Created a new feature that indicates the Age 
- &#9733; Explore data to uncover critical fraud indicators:
  -  Investigate data to identify key indicators of fraudulent activities
     - &#9642; What factors contribute to patient awareness of fraud attempts?
     - &#9642; How does the presence or absence of specific factors, like physicians, influence the occurrence of fraud?
     - &#9642; Do certain regions or healthcare facilities exhibit a higher likelihood for fraudulent activities?
     - &#9642; What common characteristics are associated with instances of fraud?
     - &#9642; Which diagnostic and procedure codes demonstrate connections to fraud?
     - &#9642; Which providers are linked to the highest frequency of fraudulent claim submissions?
     - &#9642; Is there a substantial contrast in the frequency of fraud claims between inpatient and outpatient scenarios?
- &#9733; Model Selection:
  -   Choose classification algorithms
     - &#9642; KNN
     - &#9642; Decision Tree
     - &#9642; Logistic Regression
     - &#9642; Random Forest
     
- &#9733; Data Splitting and Model Training:
  -  Divide the dataset into train, validate and test sets
     - &#9642; Train chosen models on training dataset
- &#9733; Model Evaluation:
  -   Check the performance of models on the test dataset
  - Metrics used
     - &#9642; Accuracy
     - &#9642; Precision
     - &#9642; Recall (Sensitivity or True Positive Rate)
     - &#9642; F1 Score
     - &#9642; Specificity (True Negative Rate)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **Data Dictionary** 



| Inpatient Column Name              | Outpatient Column Name               | Dtype           | Definition                  |
|------------------------------------|-------------------------------------|-----------------|-----------------------------|
| beneid                             | beneid                              | int64           | Beneficiary ID             |
| gender                             | gender                              | object          | Gender of the beneficiary  |
| race                               | race                                | object          | Race of the beneficiary    |
| renaldiseaseindicator              | renaldiseaseindicator               | object          | Indicator for renal disease|
| state                              | state                               | object          | State of the beneficiary   |
| county                             | county                              | object          | County of the beneficiary  |
| noofmonths_partacov                | noofmonths_partacov                 | int64           | Number of months with Part A coverage |
| noofmonths_partbcov                | noofmonths_partbcov                 | int64           | Number of months with Part B coverage |
| chroniccond_alzheimer              | chroniccond_alzheimer               | int64           | Chronic condition: Alzheimer's   |
| chroniccond_heartfailure           | chroniccond_heartfailure            | int64           | Chronic condition: Heart Failure |
| chroniccond_kidneydisease          | chroniccond_kidneydisease           | int64           | Chronic condition: Kidney Disease|
| chroniccond_cancer                 | chroniccond_cancer                  | int64           | Chronic condition: Cancer       |
| chroniccond_obstrpulmonary         | chroniccond_obstrpulmonary          | int64           | Chronic condition: Obstructive Pulmonary|
| chroniccond_depression             | chroniccond_depression              | int64           | Chronic condition: Depression   |
| chroniccond_diabetes               | chroniccond_diabetes                | int64           | Chronic condition: Diabetes     |
| chroniccond_ischemicheart          | chroniccond_ischemicheart           | int64           | Chronic condition: Ischemic Heart|
| chroniccond_osteoporasis           | chroniccond_osteoporasis            | int64           | Chronic condition: Osteoporosis |
| chroniccond_rheumatoidarthritis    | chroniccond_rheumatoidarthritis     | int64           | Chronic condition: Rheumatoid Arthritis|
| chroniccond_stroke                 | chroniccond_stroke                  | int64           | Chronic condition: Stroke        |
| ipannualreimbursementamt           | ipannualreimbursementamt            | float64         | Inpatient Annual Reimbursement Amount   |
| ipannualdeductibleamt              | ipannualdeductibleamt               | float64         | Inpatient Annual Deductible Amount      |
| opannualreimbursementamt           | opannualreimbursementamt            | float64         | Outpatient Annual Reimbursement Amount  |
| opannualdeductibleamt              | opannualdeductibleamt               | float64         | Outpatient Annual Deductible Amount     |
| deceased                           | deceased                            | int64           | Deceased indicator          |
| age                                | age                                 | int64           | Age of the beneficiary      |
| total_reimbursed_amt               | total_reimbursed_amt                | float64         | Total Reimbursed Amount     |
| total_deductible_amt               | total_deductible_amt                | float64         | Total Deductible Amount     |
| dob_year                           | dob_year                            | int64           | Year of birth               |
| dob_month                          | dob_month                           | int64           | Month of birth              |
| dob_day                            | dob_day                             | int64           | Day of birth                |
| race_0                             | race_0                              | int64           | Race 0                      |
| race_1                             | race_1                              | int64           | Race 1                      |
| race_2                             | race_2                              | int64           | Race 2                      |
| race_3                             | race_3                              | int64           | Race 3                      |
| claimid                            | claimid                             | int64           | Claim ID                    |
| claimstartdt                       | claimstartdt                        | object          | Claim start date            |
| claimenddt                         | claimenddt                          | object          | Claim end date              |
| provider                           | provider                            | object          | Healthcare provider         |
| inscclaimamtreimbursed             | inscclaimamtreimbursed              | float64         | Insurance claim amount reimbursed     |
| attendingphysician                 | attendingphysician                  | object          | Attending physician         |
| operatingphysician                 | operatingphysician                  | object          | Operating physician         |
| otherphysician                     | otherphysician                      | object          | Other physician             |
| admissiondt                        |                                   | object          | Admission date              |
| clmadmitdiagnosiscode              |                                   | object          | Admission diagnosis code    |
| deductibleamtpaid                  | deductibleamtpaid                   | float64         | Deductible amount paid      |
| dischargedt                        |                                   | object          | Discharge date              |
| diagnosisgroupcode                 |                                   | object          | Diagnosis group code        |
| clmdiagnosiscode_1                 | clmdiagnosiscode_1                  | object          | Claim diagnosis code 1      |
| clmdiagnosiscode_2                 | clmdiagnosiscode_2                  | object          | Claim diagnosis code 2      |
| clmdiagnosiscode_3                 | clmdiagnosiscode_3                  | object          | Claim diagnosis code 3      |
| clmdiagnosiscode_4                 | clmdiagnosiscode_4                  | object          | Claim diagnosis code 4      |
| clmdiagnosiscode_5                 | clmdiagnosiscode_5                  | object          | Claim diagnosis code 5      |
| clmdiagnosiscode_6                 | clmdiagnosiscode_6                  | object          | Claim diagnosis code 6      |
| clmdiagnosiscode_7                 | clmdiagnosiscode_7                  | object          | Claim diagnosis code 7      |
| clmdiagnosiscode_8                 | clmdiagnosiscode_8                  | object          | Claim diagnosis code 8      |
| clmdiagnosiscode_9                 | clmdiagnosiscode_9                  | object          | Claim diagnosis code 9      |
| clmdiagnosiscode_10                | clmdiagnosiscode_10                 | object          | Claim diagnosis code 10     |
| clmprocedurecode_1                 | clmprocedurecode_1                  | object          | Claim procedure code 1      |
| clmprocedurecode_2                 | clmprocedurecode_2                  | object          | Claim procedure code 2      |
| clmprocedurecode_3                 | clmprocedurecode_3                  | object          | Claim procedure code 3      |
| claimduration                      |                                   | int64           | Claim duration              |
| numphysicians                      |                                   | int64           | Number of physicians        |
| potentialfraud                     | potentialfraud                      | int64           | Potential fraud indicator   |
| potentialfraud_encoded             | potentialfraud_encoded               | int64           | Encoded potential fraud indicator  |


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **Steps to Reproduce** 

## Ordered List:
     1. Clone this repo.
     2. Acquire the data from [kaggle](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis?select=Train_Beneficiarydata-1542865627584.csv)
     3. Run data preprocessing and feature engineering scripts.
     4. Explore data using provided notebooks.
     5. Train and evaluate regression models using the provided notebook.
     6. Replicate the Healthcare Fraud Buster process using the provided instructions.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **Recommendations**

## Actionable recommendations based on project's insights:
- &#9733; Optimize 
- &#9733; Monitor healthcare provider fraud 
- &#9733; Refine 
- &#9733; Quality Testing and Validation

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **Takeaways and Conclusions**
In conclusion, our analysis has unveiled essential insights into healthcare provider fraud . The adoption of a data-driven approach not only ensures more precise quality assessments but also empowers medicare with data-backed decisions. With our current features we are able to more accurately predict inpatient than outpatient fraudulent claims.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------