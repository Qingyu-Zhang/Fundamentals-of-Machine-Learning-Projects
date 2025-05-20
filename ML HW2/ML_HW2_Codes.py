# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 13:31:36 2025

@author: Qingyu Zhang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

# Load the dataset
file_path = "techSalaries2017.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Drop one education variable and one race variable to avoid multicollinearity
df.drop(columns=['Some_College', 'Race_Hispanic'], inplace=True)

# Define the final set of predictors
predictors = [
    'yearsofexperience', 'yearsatcompany', 'Masters_Degree', 'Bachelors_Degree',
    'Doctorate_Degree', 'Highschool', 'Race_Asian', 'Race_White',
    'Race_Two_Or_More', 'Race_Black', 'Age', 'Height', 'SAT', 'GPA'
]



# Drop rows with missing values in relevant columns
df_filtered = df.dropna(subset=predictors + ['totalyearlycompensation', 'Race', 'Education'])

# ============================ #
# Question 1: Multiple Linear Regression
# ============================ #
print('Question 1:')

# Find the best predictor using single-variable regressions
best_r2 = 0
best_predictor = None

for predictor in predictors:
    X_single = df_filtered[[predictor]]
    y = df_filtered['totalyearlycompensation']
    
    model = LinearRegression()
    model.fit(X_single, y)
    
    r2 = model.score(X_single, y)
    
    if r2 > best_r2:
        best_r2 = r2
        best_predictor = predictor

# Fit multiple linear regression with all predictors
X = df_filtered[predictors]
y = df_filtered['totalyearlycompensation']

model = LinearRegression()
model.fit(X, y)
r_squared_full = model.score(X, y)

print(f"Best predictor: {best_predictor}, R²: {best_r2:.4f}")
print(f"R² (Full Model): {r_squared_full:.4f}")


# Plot best predictor regression
x_best_single=df_filtered[[best_predictor]]
best_ols=LinearRegression().fit(x_best_single, y)

plt.figure(figsize=(6, 4))
plt.scatter(x_best_single, y, alpha=0.5)
plt.plot(x_best_single, best_ols.predict(x_best_single), color='red')
plt.xlabel(best_predictor)
plt.ylabel("Total Annual Compensation")
plt.title(f"Best Predictor: {best_predictor} (R²={best_r2:.4f})")
plt.show()

# Plot actual vs. predicted values for full model
y_pred = model.predict(X)

plt.figure(figsize=(6, 4))
plt.scatter(y, y_pred, alpha=0.5)
plt.xlabel("Actual Compensation")
plt.ylabel("Predicted Compensation")
plt.title(f"Multiple Linear Regression (R²={r_squared_full:.4f})")
plt.show()

print()


# ============================ #
# Question 2: Ridge Regression
# ============================ #
print('Question 2:')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize predictors
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit Ridge Regression
alphas = np.logspace(-3, 3, 100)
ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
ridge_cv.fit(X_train_scaled, y_train)

ridge_r2_test = ridge_cv.score(X_test_scaled, y_test)

print(f"Ridge Regression - Optimal Lambda: {ridge_cv.alpha_:.4f}")
print(f"R² (Test Set): {ridge_r2_test:.4f}")

# Ridge Regression: Actual vs. Predicted Scatter Plot
y_pred_ridge = ridge_cv.predict(X_test_scaled)

plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred_ridge, alpha=0.5, color="orange")
plt.xlabel("Actual Compensation")
plt.ylabel("Predicted Compensation")
plt.title(f"Ridge Regression (R²={ridge_r2_test:.4f})")
plt.show()



print()



# ============================ #
# Question 3: Lasso Regression
# ============================ #
print('Question 3:')

lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)

lasso_r2_test = lasso_cv.score(X_test_scaled, y_test)
num_zero_coefs = np.sum(lasso_cv.coef_ == 0)


print(f"Lasso Regression - Optimal Lambda: {lasso_cv.alpha_:.4f}")
print(f"R² (Test Set): {lasso_r2_test:.4f}")
print(f"Number of Coefficients Shrunk to Zero: {num_zero_coefs}")


# Lasso Regression: Actual vs. Predicted Scatter Plot
y_pred_lasso = lasso_cv.predict(X_test_scaled)

plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred_lasso, alpha=0.5, color="green")
plt.xlabel("Actual Compensation")
plt.ylabel("Predicted Compensation")
plt.title(f"Lasso Regression (R²={lasso_r2_test:.4f})")
plt.show()



print()



# ============================ #
# Question 4: Gender Pay Gap (Logistic Regression)
# ============================ #
print('Question 4:')

# We should filter the dataframe by gender column
df_gender = df[df['gender'].isin(['Male', 'Female'])].copy()
df_gender['gender_binary'] = df_gender['gender'].map({'Male': 1, 'Female': 0})

#Build logistic regression btw the single variable total compensation and gender binary
X_gender = df_gender[['totalyearlycompensation']]
y_gender = df_gender['gender_binary']

# Train-test split
X_gender_train, X_gender_test, y_gender_train, y_gender_test = train_test_split(
    X_gender, y_gender, test_size=0.2, random_state=42
)

# Standardize predictors
scaler = StandardScaler()
X_gender_train_scaled = scaler.fit_transform(X_gender_train)
X_gender_test_scaled = scaler.transform(X_gender_test)

# Fit logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_gender_train_scaled, y_gender_train)

# Compute AUROC and AP scores on the test set
y_prob_test = log_reg.predict_proba(X_gender_test_scaled)[:, 1]
auroc_test = metrics.roc_auc_score(y_gender_test, y_prob_test)
ap_test = metrics.average_precision_score(y_gender_test, y_prob_test)

print(f"Gender From Yearly Compensation - AUROC: {auroc_test:.4f}, AP: {ap_test:.4f}")

# Plot ROC and PR curves using the test set
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
RocCurveDisplay.from_estimator(log_reg, X_gender_test_scaled, y_gender_test, ax=ax[0])
PrecisionRecallDisplay.from_estimator(log_reg, X_gender_test_scaled, y_gender_test, ax=ax[1])
ax[0].set_title("ROC Curve - Gender From Yearly Compensation")
ax[1].set_title("PR Curve - Gender From Yearly Compensation")
plt.show()




# Convert probabilities to binary predictions using 0.5 threshold
y_pred_test = (y_prob_test >= 0.5).astype(int)

# Compute Precision, Recall, and Accuracy
precision = precision_score(y_gender_test, y_pred_test)
recall = recall_score(y_gender_test, y_pred_test)
accuracy = accuracy_score(y_gender_test, y_pred_test)

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")



print(df['gender'].value_counts())




print()


# ============================ #
# Question 5: High vs. Low Salary Prediction (Logistic Regression) with Train-Test Split
# ============================ #
print('Question 5:')

# Define relevant predictors based on the problem statement
selected_predictors = ['yearsofexperience', 'Age', 'Height', 'SAT', 'GPA']

# Define outcome variable (binary high/low salary)
median_salary = df['totalyearlycompensation'].median()
df['high_salary'] = (df['totalyearlycompensation'] > median_salary).astype(int)
y_salary = df['high_salary']

# Split into training and test sets
X_salary_train, X_salary_test, y_salary_train, y_salary_test = train_test_split(
    df[selected_predictors], y_salary, test_size=0.2, random_state=42
)

# Standardize predictors
scaler = StandardScaler()
X_salary_train_scaled = scaler.fit_transform(X_salary_train)
X_salary_test_scaled = scaler.transform(X_salary_test)

# Evaluate each predictor separately using Logistic Regression
auroc_scores = {}
ap_scores = {}

for predictor in selected_predictors:
    X_train_single = X_salary_train[[predictor]]
    X_test_single = X_salary_test[[predictor]]
    
    X_train_single_scaled = scaler.fit_transform(X_train_single)
    X_test_single_scaled = scaler.transform(X_test_single)

    model = LogisticRegression()
    model.fit(X_train_single_scaled, y_salary_train)

    y_prob_test = model.predict_proba(X_test_single_scaled)[:, 1]
    auroc_scores[predictor] = metrics.roc_auc_score(y_salary_test, y_prob_test)
    ap_scores[predictor] = metrics.average_precision_score(y_salary_test, y_prob_test)

# Identify best single predictor based on AUROC
best_single_predictor = max(auroc_scores, key=auroc_scores.get)


# Fit logistic regression with all predictors
log_reg_salary = LogisticRegression(max_iter=1000)
log_reg_salary.fit(X_salary_train_scaled, y_salary_train)

# Compute AUROC and AP on the test set
y_prob_test_salary = log_reg_salary.predict_proba(X_salary_test_scaled)[:, 1]
auroc_scores["All Predictors"] = metrics.roc_auc_score(y_salary_test, y_prob_test_salary)
ap_scores["All Predictors"] = metrics.average_precision_score(y_salary_test, y_prob_test_salary)

# Convert scores into a DataFrame for better readability
results_df = pd.DataFrame({"AUROC": auroc_scores, "AP": ap_scores})

# Print results
print("AUROC and AP Scores for Each Predictor:")
print(results_df)



# Plot ROC and PR curves for the best single predictor
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
X_train_best_scaled = scaler.fit_transform(X_salary_train[[best_single_predictor]])
X_test_best_scaled = scaler.fit_transform(X_salary_test[[best_single_predictor]])
best_model = LogisticRegression()
best_model.fit(X_train_best_scaled, y_salary_train)

RocCurveDisplay.from_estimator(best_model, X_test_best_scaled, y_salary_test, ax=ax[0])
PrecisionRecallDisplay.from_estimator(best_model, X_test_best_scaled, y_salary_test, ax=ax[1])
ax[0].set_title(f"ROC Curve - Best Single Predictor ({best_single_predictor})")
ax[1].set_title(f"PR Curve - Best Single Predictor ({best_single_predictor})")
plt.show()

# Plot ROC and PR curves for all predictors combined
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
RocCurveDisplay.from_estimator(log_reg_salary, X_salary_test_scaled, y_salary_test, ax=ax[0])
PrecisionRecallDisplay.from_estimator(log_reg_salary, X_salary_test_scaled, y_salary_test, ax=ax[1])
ax[0].set_title("ROC Curve - All Predictors Combined")
ax[1].set_title("PR Curve - All Predictors Combined")
plt.show()


print()


# ============================ #
# Extra Credit: Data Distributions
# ============================ #
print('Extra Credit:')

# Plot distributions of salary, height, and age
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
variables = ['totalyearlycompensation', 'Height', 'Age']

for i, var in enumerate(variables):
    axes[i].hist(df_filtered[var], bins=50, alpha=0.7, edgecolor='black', density=True)
    axes[i].set_title(f'Distribution of {var}')

plt.tight_layout()
plt.show()




















