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
  - &#9733; Use a classification ML model for healthcare fraud detection
  - &#9733; Identify key drivers of healthcare fraud
  - &#9733; Improve fraud detection accuracy
  - &#9733; Provide insights into healthcare fraud variations
  - &#9733; Deliver a comprehensive report to the data science team responsible for addressing healthcare fraud  


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **Initial Thoughts**

###    In the initial phases of the project, data collection and preprocessing are paramount, ensuring the availability of high-quality healthcare data for analysis. This foundational stage lays the groundwork for harnessing data effectively in predicting healthcare fraud. We belive that inpatients claims have a higher likelyhood of being fraudulent. 

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **Skill Set Used **

- &#9733; Skill Set used
     - &#9642; Python
     - &#9642; NumPy
     - &#9642; Seaborn
     - &#9642; Azure
     - &#9642; Matplotlib

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## **Deliverables **

- &#9733; List of deliverables:
     - &#9642; Final Report
     - &#9642; Slide Show
     - &#9642; Front end link to dashboard

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## **Modeling **

- &#9733; List of models:
     - &#9642; Decision Tree
     - &#9642; Random Forest 
     - &#9642; KNearest Neighbor
     - &#9642; Logistic Regression 
- &#9733; Best model:
     - &#9642; TBD

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **The Plan**
- &#9733; Acquire data from https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis?select=Train-1542865627584.csv
- &#9733; Prepare data that's acquired:
  -  Feature Selection/Engineering
     - &#9642; Replace spaces with underscores
     - &#9642; Created new binary feature for fraud or legitimate claim 
     - &#9642; Created new feature that shows Patient type 
     - &#9642; Created a new feature that indicates the Age 
     - &#9642; Created a new feature that indicates the Claim diagnostic code
     - &#9642; Created a new feature that indicates the Claim procedure code
     - &#9642; Rename NoOfMonths_PartACov
     - &#9642; Rename NoOfMonths_PartBCov   
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
     - &#9642; Polynomial Regression
     - &#9642; Gradient Boosting Models(XGBoost, LightGBM)
     - &#9642; AdaBoost
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



| Column Name               | Dtype           | Definition                                            |
|---------------------------|-----------------|-------------------------------------------------------|
| **ClaimID**               | object          | Claim ID                                              |
| **Provider**              | object          | Healthcare provider's ID                              |
| **InscClaimAmtReimbursed**| int64           | Amount reimbursed by insurance for the claim         |
| **AttendingPhysician**    | object          | Attending physician for the claim                     |
| **OperatingPhysician**    | object          | Operating physician for the claim                     |
| **OtherPhysician**        | object          | Other physician for the claim                         |
| **AdmissionDt**           | object          | Date of admission                                     |
| **ClmAdmitDiagnosisCode** | object          | Diagnosis code for admission                          |
| **DeductibleAmtPaid**     | float64         | Amount paid as deductible                             |
| **DiagnosisGroupCode**    | object          | Group code for diagnoses                              |
| **ClmDiagnosisCode_1**    | object          | Diagnosis code 1 for the claim                        |
| **ClmDiagnosisCode_2**    | object          | Diagnosis code 2 for the claim                        |
| **ClmDiagnosisCode_3**    | object          | Diagnosis code 3 for the claim                        |
| **ClmDiagnosisCode_4**    | object          | Diagnosis code 4 for the claim                        |
| **ClmDiagnosisCode_5**    | object          | Diagnosis code 5 for the claim                        |
| **ClmDiagnosisCode_6**    | object          | Diagnosis code 6 for the claim                        |
| **ClmDiagnosisCode_7**    | object          | Diagnosis code 7 for the claim                        |
| **ClmDiagnosisCode_8**    | object          | Diagnosis code 8 for the claim                        |
| **ClmDiagnosisCode_9**    | object          | Diagnosis code 9 for the claim                        |
| **ClmDiagnosisCode_10**   | object          | Diagnosis code 10 for the claim                       |
| **ClmProcedureCode_1**    | float64         | Procedure code 1 for the claim                        |
| **ClmProcedureCode_2**    | float64         | Procedure code 2 for the claim                        |
| **ClmProcedureCode_3**    | float64         | Procedure code 3 for the claim                        |
| **ClmProcedureCode_4**    | float64         | Procedure code 4 for the claim                        |
| **ClmProcedureCode_5**    | float64         | Procedure code 5 for the claim                        |
| **ClmProcedureCode_6**    | float64         | Procedure code 6 for the claim                        |


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **Steps to Reproduce** 

## Ordered List:
     1. Clone this repo.
     2. Acquire the data from [kaggle](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis?select=Train_Beneficiarydata-1542865627584.csv)
     3. Run data preprocessing and feature engineering scripts.
     4. Explore data using provided notebooks.
     5. Train and evaluate classification models using the provided notebook.
     6. Replicate the potential fraud assessment process using the provided instructions.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## **Recommendations**

## Actionable recommendations based on project's insights:
- &#9733; Implement more advanced machine learning models to predict potentially fraudulent claims.
- &#9733; Take strict legal actions against fraud. 
- &#9733; Establish thorough documentation and audit trails for healthcare transactions.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## **Takeaways and Conclusions**
In conclusion, we are able to predict provider fraud for both inpatient and outpatient claims. Our analysis has unveiled essential insights into healthcare provider fraud. The adoption of a data-driven approach not only ensures more precise quality assessments but also empowers medicare with data-backed decisions.

## **Next Steps**
 - Continuously evaluate and update fraud prevention strategies.
 - Enhance fraud detection models and algorithms.
 - Promote whistleblower programs and legal actions.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------