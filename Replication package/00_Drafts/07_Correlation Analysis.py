# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:08:55 2023

@author: goodhao
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Part 1: Data Reading
# =============================================================================
# Define the file paths for data input
folder_path = r'07_Other_Data\02_Aggregated'
file_name1 = r'01 - Aggregated_Data_Selected.xlsx'
file_name2 = r'03 - Aggregated_Data_Selected.xlsx'

file_path1 = os.path.join(folder_path, file_name1)
file_path2 = os.path.join(folder_path, file_name2)

# Read the Excel files
df1 = pd.read_excel(file_path1)
df2 = pd.read_excel(file_path2)
# =============================================================================


# Part 2: Correlation Analysis and Visualization
# =============================================================================
# The EUI used here corresponds to the yearly EUI from 2008 to 2021.
# Compute the correlation matrix for the dataset (excluding the 'Year' column).
correlation_matrix = df1.drop(columns=['Year']).corr()

# Visualize the correlation matrix as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

# Update (12/03): Use shortened variable names, including the current year's EUI and previous year's EUI.
# Time range updated to 2009-2021.
# =========================
folder_path3 = r'07_Other_Data\03_Standard'
file_name3 = r'06 - Aggregated_Data_Standard_Dummy_MoreEUI.xlsx'
file_path3 = os.path.join(folder_path3, file_name3)

df4_EUIadded = pd.read_excel(file_path3)
# Estimate the correlation matrix
correlation_matrix_EUIadded = df1.drop(columns=['Year']).corr()

# Visualize the updated correlation matrix
plt.figure(figsize=(12, 8))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
heatmap = sns.heatmap(df4_EUIadded.corr(), annot=True, cmap='coolwarm', fmt=".2f")
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14)
plt.title("Correlation Matrix of Variables", fontsize=16)
plt.xticks(rotation=45)
plt.savefig(os.path.join(folder_path, 'figs', 'Correlation Matrix.png'), bbox_inches='tight', dpi=350)

# =========================


# Plot scatterplots to compare specific variables with the target variable
columns_to_compare = ['Year', 'Average EUI(kWh/m².yr)', 'Average Electricity Tariffs($/MW·h)', 
                      'Average temperature(℃)', 'Maximum temperature(℃)', 'Core CPI(%)', 
                      'GDP(current US$)', 'Industry(current US$)']
ycolumn_name = 'Increased Green Buildings'

plt.figure(figsize=(15, 20))
for i, column in enumerate(columns_to_compare, 1):
    plt.subplot(4, 2, i)
    sns.scatterplot(data=df1, x=column, y=df1[ycolumn_name], marker='*', size=df1[ycolumn_name]*2, sizes=(100, 200), legend=False)
    plt.xlabel(column)
    plt.ylabel(ycolumn_name)
plt.savefig(os.path.join(folder_path, 'figs', 'Compared_Variables3.png'), bbox_inches='tight', dpi=350)
# =============================================================================


# Part 3: Multicollinearity Check (Variance Inflation Factor - VIF)
# =============================================================================
# Exclude the dependent variable to calculate VIF for independent variables
independent_variables_vif = df1.drop(columns=['Increased Green Buildings'])

# Calculate VIF for each independent variable
vif_data = pd.DataFrame()
vif_data['Variable'] = independent_variables_vif.columns
vif_data['VIF'] = [variance_inflation_factor(independent_variables_vif.values, i) for i in range(len(independent_variables_vif.columns))]

# Note: High VIF (>10) indicates multicollinearity issues, requiring further inspection.
# =============================================================================


# Part 4: Fixed-Effects Model and Elastic Net Regression
# =============================================================================
# Use time fixed-effects by removing 2008 dummy variable to avoid dummy variable trap
data = df1.set_index('Year')
y = data['Increased Green Buildings']
X = data.drop(columns=['Increased Green Buildings'])

# Fixed-effects transformation: Remove entity-level averages
X_fe = X - X.mean()
y_fe = y - y.mean()

# Standardize the fixed-effects transformed data (Z-score normalization)
scaler = StandardScaler()
X_fe_scaled = scaler.fit_transform(X_fe)

# Save standardized data; manually add fixed-effects dummy variables for 2009-2021.
X_fe_scaled_df = pd.DataFrame(X_fe_scaled, columns=X.columns, index=X.index)
y_fe_df = pd.DataFrame(y_fe, columns=['Increased Green Buildings'], index=y.index)
df3_standard = pd.concat([y_fe_df, X_fe_scaled_df], axis=1)
folder_path3 = r'07_Other_Data\03_Standard'
file_name3 = r'04 - Aggregated_Data_Standard.xlsx'
file_path3 = os.path.join(folder_path3, file_name3)
df3_standard.to_excel(file_path3)

# Reload processed data with dummy variables added (variable names shortened for simplicity)
file_name4 = r'05 - Aggregated_Data_Standard_Dummy.xlsx'
file_path4 = os.path.join(folder_path3, file_name4)
df4 = pd.read_excel(file_path4)

# Update (12/03): Manually update variable names (e.g., 'Increased Green Buildings' -> 'GBN')
y = df4['GBN']
X = df4.drop(columns=['GBN'])

# Elastic Net parameter grid for cross-validation
param_grid = {
    'alpha': [0.1, 0.5, 1, 5, 10, 50, 100],
    'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
}

# Cross-validation to find the best parameters
elastic_net = ElasticNet()
grid_search = GridSearchCV(estimator=elastic_net, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
# =============================================================================

# Model Validation and Evaluation
# =============================================================================
# Fit the Elastic Net model
grid_search.fit(X, y)

# Retrieve best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Extract coefficients and intercept
coefficients = best_model.coef_
intercept = best_model.intercept_

# Link coefficients to variable names
feature_names = X.columns
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Visualize coefficients
plt.figure(figsize=(12, 8))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

coef_df.sort_values(by='Coefficient', ascending=False, inplace=True)
plt.bar(coef_df['Feature'], coef_df['Coefficient'])
plt.xlabel('Variables', fontsize=14)
plt.ylabel('Coefficient', fontsize=14)
plt.title('Fixed effects of variables', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.savefig(os.path.join(folder_path, 'figs', 'Fixed effects of variables.png'), bbox_inches='tight', dpi=350)

# Compare actual vs predicted values
plt.figure(figsize=(8, 5))
y_pred = best_model.predict(X)

plt.scatter(y, y_pred)
plt.xlabel('Actual changes in the number of green buildings (relative to the average)', fontsize=14)
plt.ylabel('Predicted changes (relative to the average)', fontsize=14)
plt.title('Actual vs Predicted Values', fontsize=16)
plt.savefig(os.path.join(folder_path, 'figs', 'Predicted result of fixed effects.png'), bbox_inches='tight', dpi=350)
plt.show()

# Compute performance metrics
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

# Residual analysis
plt.figure(figsize=(8, 5))
residuals = y - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted changes in the number of green buildings (relative to the average)', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.axhline(y=0, color='red', linestyle='--')
plt.ylim(-38, 38)  # Set y-axis limits for better visualization of residuals
plt.savefig(os.path.join(folder_path, 'figs', 'Residuals of prediction.png'), bbox_inches='tight', dpi=350)
plt.show()

# Cross-validation best score
best_score = grid_search.best_score_
print("Best Cross-Validation Score:", best_score)
# =============================================================================


# Part 5: Data Output (added on 2024.03.09)
# =============================================================================
# Save the coefficients of the Elastic Net model to an Excel file
file_name5 = r'04 - coef for elastic net model.xlsx'
coef_df.to_excel(os.path.join(folder_path, file_name5), index=False)
# =============================================================================

