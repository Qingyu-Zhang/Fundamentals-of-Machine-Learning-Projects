# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:42:51 2025

@author: Qingyu Zhang
"""

#Fundamentals of ML HW1
#Qingyu Zhang (Andy)
#N-number: 19903322
#NetID: qz2247


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load the dataset 
housing_data = pd.read_csv("housingUnits.csv")




#%% Question1
# Compute correlation between number of rooms / number of bedrooms / population / number of households with median_house_value
correlation_matrix1 = housing_data[['total_rooms', 'total_bedrooms', 'population', 'households', 'median_house_value']].corr()

# Visualize correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix1, annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.show()





#%% Question2
print("Question2:")
# Standardizing total_rooms and total_bedrooms by population and households
housing_data["rooms_per_person"] = housing_data["total_rooms"] / housing_data["population"]
housing_data["rooms_per_household"] = housing_data["total_rooms"] / housing_data["households"]
housing_data["bedrooms_per_person"] = housing_data["total_bedrooms"] / housing_data["population"]
housing_data["bedrooms_per_household"] = housing_data["total_bedrooms"] / housing_data["households"]

# Compute correlation between rooms/bedrooms per person vs per household with median_house_value
correlation_matrix2 = housing_data[["rooms_per_person", "rooms_per_household", "bedrooms_per_person", "bedrooms_per_household", "median_house_value"]].corr()

# Visualize correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix2, annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Choosing the better standardization approach based on correlation coefficients
best_rooms_standardization = "rooms per household" if abs(correlation_matrix2.loc["rooms_per_household", "median_house_value"]) > abs(correlation_matrix2.loc["rooms_per_person", "median_house_value"]) else "rooms per person"
best_bedrooms_standardization = "bedrooms per household" if abs(correlation_matrix2.loc["bedrooms_per_household", "median_house_value"]) > abs(correlation_matrix2.loc["bedrooms_per_person", "median_house_value"]) else "bedrooms per person"

print(f'The answer is: "{best_rooms_standardization}" and "{best_bedrooms_standardization}".')

print()




#%% Question3
print('Question3:')
# Perform simple linear regression for each predictor against median_house_value
predictors = ["housing_median_age", "rooms_per_person", "bedrooms_per_person", "population", "households", "median_income", "ocean_proximity"]
target = "median_house_value"

# Store R^2 values
r_squared_values = {}

for predictor in predictors:
    X = housing_data[[predictor]].values   # make sure it's a 2D array, which is required by LinearRegression package
    y = housing_data[target].values
    
    model = LinearRegression()
    model.fit(X, y)
    r_squared = model.score(X, y)  # Coefficient of determination
    r_squared_values[predictor] = r_squared

# Find most and least predictive variables
most_predictive = max(r_squared_values, key=lambda k: r_squared_values[k])
least_predictive = min(r_squared_values, key=r_squared_values.get)  # try different ways

# Display results
print('R-squared values are:', r_squared_values,'\n')
print(f'The most predictive predictor is "{most_predictive}", the least predictive predictor is "{least_predictive}".')


# Plot scatter plots and inspect the limitation on the data
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Median Income vs. Median House Value
axes[0].plot(housing_data["median_income"], housing_data["median_house_value"], 'o')
axes[0].set_title("Median Income vs Median House Value")
axes[0].set_xlabel("Median Income (in $1000s)")
axes[0].set_ylabel("Median House Value ($)")

# Population vs. Median House Value
axes[1].plot(housing_data["population"], housing_data["median_house_value"], 'o')
axes[1].set_title("Population vs Median House Value")
axes[1].set_xlabel("Population")
axes[1].set_ylabel("Median House Value ($)")

plt.tight_layout()
plt.show()

print()




#%% Question4
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
print('Question4:')

# Prepare data for multiple regression
X_full = housing_data[predictors].values
y = housing_data[target].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

# Train multiple regression model
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)

# y_pred_multi = multi_model.predict(X_test)
# r2_multi = r2_score(y_test, y_pred_multi)

r2_multi = multi_model.score(X_test, y_test)

# Train single predictor model (median_income only)
X_income = housing_data[["median_income"]].values
X_train_income, X_test_income, y_train_income, y_test_income = train_test_split(X_income, y, test_size=0.2, random_state=42)
income_model = LinearRegression()
income_model.fit(X_train_income, y_train_income)

# y_pred_income = income_model.predict(X_test_income)
# r2_income = r2_score(y_test_income, y_pred_income)

r2_income = income_model.score(X_test_income, y_test_income)

# Compare R^2 values
print(f'R-Squre of the multiple regression model is: {r2_multi}; R-Squre of the simple best predictor model is {r2_income}')

print()




#%% Question5
print("Question5:")
# Compute correlation between rooms_per_person and bedrooms_per_person
collinearity_rooms_bedrooms = housing_data[["rooms_per_person", "bedrooms_per_person"]].corr().iloc[0, 1]

# Compute correlation between population and households
collinearity_population_households = housing_data[["population", "households"]].corr().iloc[0, 1]

print(f'Correlation btw variable 2 and 3: {collinearity_rooms_bedrooms}; correlation btw variable 4 and 5: {collinearity_population_households}')






#%% Extra Credit
# Plot histograms for each variable to visualize their distributions
numerical_columns = ["housing_median_age", "total_rooms", "total_bedrooms", "population", 
                     "households", "median_income", "ocean_proximity", "median_house_value"]


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(numerical_columns):
    axes[i].hist(housing_data[col], bins=50)
    axes[i].set_title(col)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
