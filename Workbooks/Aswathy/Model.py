import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

###############################################################################################################
# Implementing Logistic Regression, Random Forest, and
#  K-Nearest Neighbors (KNN) 
# classification algorithms using Python's scikit-learn library

# Logistic Regression
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)
logreg_y_val_pred = logreg_model.predict(X_val)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_y_val_pred = rf_model.predict(X_val)

# K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_y_val_pred = knn_model.predict(X_val)

# Calculate evaluation metrics for each model
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("confusion_matrix:")
    print("---------------")
    print(confusion_matrix(y_true, y_pred))
    print("=" * 50)

# Evaluate models
evaluate_model(y_val, logreg_y_val_pred, "Logistic Regression")
evaluate_model(y_val, rf_y_val_pred, "Random Forest")
evaluate_model(y_val, knn_y_val_pred, "K-Nearest Neighbors")

# Feature Importance Analysis using Random Forest
rf_model.fit(X_train, y_train)
feature_importances = rf_model.feature_importances_

# Create a DataFrame to display feature importances along with their corresponding column names
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sort the DataFrame by Importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the sorted feature importances
print("\nFeature Importance Analysis:")
print(feature_importance_df)

# Plotting Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature')
plt.title("Feature Importance Analysis")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

###############################################################################################################
# pip install xgboost
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_model = XGBClassifier(objective='binary:logistic', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_y_val_pred = xgb_model.predict(X_val)
# Evaluate the XGBoost model
accuracy = accuracy_score(y_val, xgb_y_val_pred)
print(f"Accuracy: {accuracy:.4f}")
###############################################################

# Tune hyperparameters of the XGBoost model
from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5]
}

grid_search = GridSearchCV(estimator=XGBClassifier(objective='binary:logistic', random_state=42),
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=3)
grid_search.fit(X_train, y_train)

best_xgb_model = grid_search.best_estimator_
best_xgb_y_val_pred = best_xgb_model.predict(X_val)

best_accuracy = accuracy_score(y_val, best_xgb_y_val_pred)
print(f"Best Model Accuracy: {best_accuracy:.4f}")
print("Best Model Parameters:", grid_search.best_params_)

####################################################

#  Ensemble methods: Bagging (e.g., Random Forest), 
# Boosting (e.g., Gradient Boosting), and Stacking

from sklearn.ensemble import VotingClassifier

# Define the ensemble of models
ensemble_model = VotingClassifier(estimators=[
    ('Logistic Regression', logreg_model),
    ('Random Forest', rf_model),
    ('K-Nearest Neighbors', knn_model)
])

# Fit the ensemble model
ensemble_model.fit(X_train, y_train)

# Make predictions using the ensemble model
ensemble_y_val_pred = ensemble_model.predict(X_val)

# Evaluate the ensemble model
evaluate_model(y_val, ensemble_y_val_pred, "Ensemble Model")

####################################################

# AdaBoost (Adaptive Boosting)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
base_model = DecisionTreeClassifier(max_depth=1)  # The weak classifier (often a decision stump)
adaboost_model = AdaBoostClassifier(base_model, n_estimators=50, random_state=42)
adaboost_model.fit(X_train, y_train)
adaboost_y_val_pred = adaboost_model.predict(X_val)
accuracy = accuracy_score(y_val, adaboost_y_val_pred)
print(f"Accuracy: {accuracy:.4f}")
####################################################

# Support Vector Machines (SVM) 

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)
svm_y_val_pred = svm_model.predict(X_val)
accuracy = accuracy_score(y_val, svm_y_val_pred)
print(f"Accuracy: {accuracy:.4f}")


####################################################
# Baseline Model of SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_y_val_pred = svm_model.predict(X_val)
accuracy = accuracy_score(y_val, svm_y_val_pred)
print(f"Accuracy: {accuracy:.4f}")
####################################################

# LightGBM model
# pip install lightgbm
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(X_train, y_train)

lgb_y_val_pred = lgb_model.predict(X_val)
accuracy = accuracy_score(y_val, lgb_y_val_pred)
print(f"LightGBM Baseline Model Accuracy: {accuracy:.4f}")
####################################################
# LightGBM model with hyperparameter tuning
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = { 
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(estimator=lgb.LGBMClassifier(random_state=42),
                            param_grid=param_grid,
                            scoring='accuracy',
                            cv=3)
grid_search.fit(X_train, y_train)

best_lgb_model = grid_search.best_estimator_
best_lgb_y_val_pred = best_lgb_model.predict(X_val)

best_accuracy = accuracy_score(y_val, best_lgb_y_val_pred)
print(f"Best LightGBM Model Accuracy: {best_accuracy:.4f}")




