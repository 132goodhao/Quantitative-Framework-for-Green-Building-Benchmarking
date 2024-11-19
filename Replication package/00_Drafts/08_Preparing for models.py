# -*- coding: utf-8 -*-
"""
Pre-matching preparations
@author: goodhao
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportions_chisquare

# Data preprocessing for IPW and Propensity Score Estimation

# Part 1: Data Reading
# =============================================================================
folder_path = r'08_Preparing for models'
file_name1 = r'02 - Energy Performance Data from 2016 to 2021_for_IPW_out of blank_variables selected.xlsx'
file_name2 = r'03 - Energy Performance Data from 2016 to 2021_for_IPW_OneHot1.xlsx'
file_name3 = r'04 - Energy Performance Data from 2016 to 2021_for_IPW_Standardized.xlsx'
file_path1 = os.path.join(folder_path, file_name1)
file_path2 = os.path.join(folder_path, file_name2)
file_path3 = os.path.join(folder_path, file_name3)
# Read Excel files
df1 = pd.read_excel(file_path1)
# =============================================================================

# Part 2: One-Hot Encoding and Binary Encoding
# =============================================================================
# Process the categorical variable w: Green Mark Rating
df1['Green Labeling'] = df1['Green Mark Rating'].notnull().astype(int)

# Process the remaining categorical covariates x: Type, Type of ACS, Size, PV
# Using One-Hot Encoding
encoder = OneHotEncoder(sparse=False)
encoded_types = encoder.fit_transform(df1[['Type', 'Type of ACS']])
encoded_type_columns = encoder.get_feature_names_out(['Type', 'Type of ACS'])
df_encoded_types = pd.DataFrame(encoded_types, columns=encoded_type_columns)
# Using Binary Encoding
df1['Size Encoded'] = df1['Size'].map({'Large': 1, 'Small': 0})
df1['PV Encoded'] = df1['PV'].map({'Yes': 1, 'No': 0})
# Concatenate data
df_combined = pd.concat([df1, df_encoded_types], axis=1)

# Output the concatenated data to an Excel file
df_combined.to_excel(file_path2, index=False)
# =============================================================================

# Part 3: Remove Sparse Variables, Data Standardization, Z-score
# =============================================================================

# Count the values and frequencies of variables to check for sparse variables
# Define variable names
variable_distri_names = ['Type', 'Type of ACS', 'Size', 'PV']
# Define the output file name
filename_variable_distribution = os.path.join(folder_path, 'data_distri', 'variable_distribution_for_IPW.xlsx')
# Define a function to count variable frequencies
def variable_distribution(df, variable_names):
    distributions = {}  # Store distribution results for each variable
    for variable_name in variable_distri_names:
        variable_counts = (df_combined[variable_name]
                          .value_counts()
                          .reset_index()
                          .set_axis([variable_name, 'Frequency'], axis=1)  # Rename the columns directly
                          .sort_values(by=variable_name, ascending=False)
                          .reset_index(drop=True))
        distributions[variable_name] = variable_counts
    return distributions

# Define the output function
def save_variable_distribution_to_excel(variable_distri, file_name):
    # Use ExcelWriter to write to an Excel file
    with pd.ExcelWriter(file_name) as writer:
        for variable_name, counts in variable_distri.items():
            safe_sheet_name = variable_name.replace('/', '-')
            # Write each DataFrame to a different sheet with the sheet name as the variable name
            counts.to_excel(writer, sheet_name=safe_sheet_name)

# Output the dictionary of variable frequencies
variable_distri = variable_distribution(df_combined, variable_distri_names)
# Output the Excel file with variable frequency statistics
save_variable_distribution_to_excel(variable_distri, filename_variable_distribution)

# Function to remove sparse variables
def remove_sparse_columns(df, threshold=0.95):
    columns_to_drop = []
    for column in df.columns:
        top_freq = df[column].value_counts(normalize=True).iloc[0]
        if top_freq > threshold:
            columns_to_drop.append(column)
    return columns_to_drop

# Merge sparse variables with those already marked for dropping
dropped_columns_for_sparse = remove_sparse_columns(df_combined)
dropped_columns_for_repeating = ['Building Name', 'Address', 'Green Mark Rating', 'Type', 'Type of ACS', 'Size', 'PV']
df_dropped_columns = dropped_columns_for_sparse + dropped_columns_for_repeating
# Create a compact matrix
df_combined_tight = df_combined.drop(columns=df_dropped_columns)

# Identify variables that need to be standardized
non_binary_columns = df_combined_tight.columns[(df_combined_tight.nunique() > 2)]
# Apply standardization
scaler = StandardScaler()
df_combined_tight[non_binary_columns] = scaler.fit_transform(df_combined_tight[non_binary_columns])
# Output the standardized data to an Excel file
df_combined_tight.to_excel(file_path3, index=False)
# =============================================================================

# Part 4: Logistic Regression
# =============================================================================
# Construct independent variables, excluding all EUI-related items and Green Labeling
column_names = df_combined_tight.columns.tolist()
excluded_columns = [col for col in column_names if 'EUI' in col]
excluded_columns.append('Green Labeling')
X = df_combined_tight.drop(columns=excluded_columns)
X_constant = sm.add_constant(X)  # Add a constant term

# Construct the dependent variable
y = df_combined_tight['Green Labeling']

# Build the model
logit_model_forAll = sm.Logit(y, X_constant).fit()
logit_model_summary_forAll = logit_model_forAll.summary()
logit_model_summary_forAll

# Identify variables with large p-values
p_values = logit_model_forAll.pvalues
variables_with_high_p_values_except_const = p_values[(p_values > 0.7) & (p_values.index != 'const')].index.tolist()

# Perform logistic regression again
X_constant_Pcontrolled = X_constant.drop(columns=variables_with_high_p_values_except_const)
logit_model_Pcontrolled = sm.Logit(y, X_constant_Pcontrolled).fit()
logit_model_summary_Pcontrolled = logit_model_Pcontrolled.summary()
logit_model_summary_Pcontrolled

# Output the results of both regressions
file_path4 = os.path.join(folder_path, 'data_distri')
file_name_logitmodel_forAll = os.path.join(file_path4, 'Results of logit Model for All.xlsx')
file_name_logitmodel_Pcontrolled = os.path.join(file_path4, 'Results of logit Model P value controlled.xlsx')
# Save the logistic regression results to a DataFrame
def results_toDataFrame(logit_model):
    results_df = pd.DataFrame({
        'Coefficients': logit_model.params,
        'Standard Errors': logit_model.bse,
        'z-values': logit_model.tvalues,
        'P-values': logit_model.pvalues
    })
    return results_df
results_df_forALL = results_toDataFrame(logit_model_forAll)
results_df_Pcontrolled = results_toDataFrame(logit_model_Pcontrolled)

results_df_forALL.to_excel(file_name_logitmodel_forAll)
results_df_Pcontrolled.to_excel(file_name_logitmodel_Pcontrolled)
# =============================================================================

# Part 5: Propensity Score Estimation
# =============================================================================
# Reload df_combined_tight
df_combined_tight = pd.read_excel(file_path3)

# Estimate the propensity score
propensity_scores = logit_model_Pcontrolled.predict(X_constant_Pcontrolled)
df_combined_tight['propensity_score'] = propensity_scores

# Define the treatment variable, in this case, Green Labeling
treatment = df_combined_tight['Green Labeling']
# Separate treatment and control groups
treatment_group = df_combined_tight[treatment == 1]
control_group = df_combined_tight[treatment == 0]

# Initialize NearestNeighbors models
nn_model_C = NearestNeighbors(n_neighbors=3, algorithm='auto')
nn_model_T = NearestNeighbors(n_neighbors=3, algorithm='auto')

# Train the model on the control group using propensity scores
nn_model_C.fit(np.array(control_group['propensity_score']).reshape(-1, 1))

# Train the model on the treatment group using propensity scores
nn_model_T.fit(np.array(treatment_group['propensity_score']).reshape(-1, 1))

# Find the nearest 3 neighbors for each treatment group member from the control group
distances_treatment, indices_treatment = nn_model_C.kneighbors(np.array(treatment_group['propensity_score']).reshape(-1, 1))

# Find the nearest 3 neighbors for each control group member from the treatment group
distances_control, indices_control = nn_model_T.kneighbors(np.array(control_group['propensity_score']).reshape(-1, 1))

# Convert indices to those of the original dataset
indices_treatment_transformed = control_group.iloc[indices_treatment.flatten()].index.values.reshape(indices_treatment.shape)
indices_control_transformed = treatment_group.iloc[indices_control.flatten()].index.values.reshape(indices_control.shape)

# Add indices to the original dataset
df_PScore_and_Indices = df_combined_tight
for i in range(3):
    # Add neighbor indices for the treatment group
    df_PScore_and_Indices.loc[treatment_group.index, f'neighbor_indices_{i+1}'] = indices_treatment_transformed[:, i]
    
    # Add neighbor indices for the control group
    df_PScore_and_Indices.loc[control_group.index, f'neighbor_indices_{i+1}'] = indices_control_transformed[:, i]

# Output data
# Restore EUI normalization
excluded_columns.remove('Green Labeling')
df_PScore_and_Indices[excluded_columns] = df1[excluded_columns]
file_name4 = r'05 - Energy Performance Data from 2016 to 2021_for_IPW_Pscore.xlsx'
file_path4 = os.path.join(folder_path, file_name4)
df_PScore_and_Indices.to_excel(file_path4, index=False)
# =============================================================================

# Part 6: Balance Test
# =============================================================================
# Visualize the overlap of propensity scores between groups
file_path5 = os.path.join(folder_path, 'figs')
sns.kdeplot(treatment_group['propensity_score'], label='Treatment Group', common_norm=True, bw='silverman')
sns.kdeplot(control_group['propensity_score'], label='Control Group', common_norm=True, bw='silverman')
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.xlim(0, 1)
plt.legend()
plt.savefig(os.path.join(file_path5, 'Frequency coverage of Pscore' + '.png'), bbox_inches='tight', dpi=350)

# Test balance on all variables except Green Labeling and Propensity Score
variables_to_check = df_combined_tight.columns.drop(['Green Labeling', 'propensity_score', 
                                                     'neighbor_indices_1', 'neighbor_indices_2', 'neighbor_indices_3'])

# Initialize a dictionary to store the results
balance_checks = {
    'variable': [],
    'SMD': [],
    't_test_p_value': [],
    'variance_ratio': [],
    'chi2_p_value': []
}

# Perform balance checks for each variable
for var in variables_to_check:
    treatment_group_var = treatment_group[var]
    control_group_var = control_group[var]

    # Calculate Standardized Mean Difference (SMD)
    mean_diff = treatment_group_var.mean() - control_group_var.mean()
    pooled_std = np.sqrt((treatment_group_var.std()**2 + control_group_var.std()**2) / 2)
    SMD = mean_diff / pooled_std

    # T-test
    t_test_p_value = ttest_ind(treatment_group_var.dropna(), control_group_var.dropna()).pvalue

    # Variance ratio
    variance_ratio = treatment_group_var.var() / control_group_var.var()

    # Chi-square test or Fisher's exact test (for categorical variables)
    # Assuming the categorical variable has values 0 or 1
    if df_combined_tight[var].nunique() == 2:
        table = pd.crosstab(df_combined_tight['Green Labeling'], df_combined_tight[var])
        chi2_p_value = proportions_chisquare(table.iloc[0], table.iloc[1])[1]
    else:
        chi2_p_value = None

    # Save the results
    balance_checks['variable'].append(var)
    balance_checks['SMD'].append(SMD)
    balance_checks['t_test_p_value'].append(t_test_p_value)
    balance_checks['variance_ratio'].append(variance_ratio)
    balance_checks['chi2_p_value'].append(chi2_p_value)

# Output the balance test results
balance_checks_df = pd.DataFrame(balance_checks)
balance_checks_df.to_excel(os.path.join(folder_path, 'data_distri', 'Balance test for SMD-T_test-Variance_ratio-Chi2_p_value.xlsx'))
# =============================================================================