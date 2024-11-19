# -*- coding: utf-8 -*-
"""
Matching and post-matching
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportions_chisquare


# Evaluate Treatment Effect using Propensity Score

# Part 1: Data Reading
# =============================================================================
folder_path = r'09_Treatment Effect for Propensity Score'
# Original file
file_name1 = r'00 - Energy Performance Data from 2016 to 2021_Pscore.xlsx'
# ATET
file_name2 = r'01 - Energy Performance Data from 2016 to 2021_Pscore_treatment group.xlsx'
file_name3 = r'02 - Energy Performance Data from 2016 to 2021_Pscore_average control group.xlsx'
file_name4 = r'03 - Energy Performance Data from 2016 to 2021_Pscore_post match ATET.xlsx'
# ATENT
file_name5 = r'04 - Energy Performance Data from 2016 to 2021_Pscore_control group.xlsx'
file_name6 = r'05 - Energy Performance Data from 2016 to 2021_Pscore_average treatment group.xlsx'
file_name7 = r'06 - Energy Performance Data from 2016 to 2021_Pscore_post match ATENT.xlsx'
# File paths
file_path1 = os.path.join(folder_path, file_name1)
file_path2 = os.path.join(folder_path, file_name2)
file_path3 = os.path.join(folder_path, file_name3)
file_path4 = os.path.join(folder_path, file_name4)
file_path5 = os.path.join(folder_path, file_name5)
file_path6 = os.path.join(folder_path, file_name6)
file_path7 = os.path.join(folder_path, file_name7)
# Read Excel files
df1 = pd.read_excel(file_path1)
# =============================================================================

# Part 2: Generate Matched Samples
# =============================================================================
# Select treatment group samples for calculating ATET
treatment_group = df1[df1['Green Labeling'] == 1]
# Initialize DataFrame to store average control group data
average_control_group = pd.DataFrame()

# Calculate the average of matched control group samples for each treatment group sample

# for idx, row in treatment_group.iterrows():
#     matched_indices = [row['neighbor_indices_1'], row['neighbor_indices_2'], row['neighbor_indices_3']]
#     matched_controls = df1.loc[matched_indices]
#     average_control = matched_controls.mean()
#     if isinstance(average_control, pd.Series):
#         average_control = average_control.to_frame().T
#     average_control_group = pd.concat([average_control_group, average_control], ignore_index=True)

df_post_match_control_group_ATET = pd.concat([treatment_group.reset_index(drop=True), average_control_group.reset_index(drop=True)], axis=1)

# Select control group samples for calculating ATENT
control_group = df1[df1['Green Labeling'] == 0]
# Initialize DataFrame to store average treatment group data
average_treatment_group = pd.DataFrame()

# Calculate the average of matched treatment group samples for each control group sample
# for idx, row in control_group.iterrows():
#     matched_indices = [row['neighbor_indices_1'], row['neighbor_indices_2'], row['neighbor_indices_3']]
#     matched_treatments = df1.loc[matched_indices]
#     average_treatment = matched_treatments.mean()
#     if isinstance(average_treatment, pd.Series):
#         average_treatment = average_treatment.to_frame().T
#     average_treatment_group = pd.concat([average_treatment_group, average_treatment], ignore_index=True)

df_post_match_treatment_group_ATENT = pd.concat([control_group.reset_index(drop=True), average_treatment_group.reset_index(drop=True)], axis=1)
    
# Simplify the process using a function to calculate the average treatment and control groups
# ====================    
def calculate_average_matched_group(group_df, full_df, neighbor_cols):

    average_group = pd.DataFrame()

    for idx, row in group_df.iterrows():
        matched_indices = [row[col] for col in neighbor_cols]
        matched_samples = full_df.loc[matched_indices]
        average_sample = matched_samples.mean()
        if isinstance(average_sample, pd.Series):
            average_sample = average_sample.to_frame().T
        average_group = pd.concat([average_group, average_sample], ignore_index=True)

    return average_group
# ====================
    
neighbor_cols = ['neighbor_indices_1', 'neighbor_indices_2', 'neighbor_indices_3']  
# Apply the function to the treatment group
average_control_group = calculate_average_matched_group(treatment_group, df1, neighbor_cols) 
average_treatment_group = calculate_average_matched_group(control_group, df1, neighbor_cols)

# Data output
# ATET
treatment_group.to_excel(file_path2)
average_control_group.to_excel(file_path3)
df_post_match_control_group_ATET.to_excel(file_path4)
# ATENT
control_group.to_excel(file_path5)
average_treatment_group.to_excel(file_path6)
df_post_match_treatment_group_ATENT.to_excel(file_path7)
# =============================================================================

# Part 3: Comparison Before and After Matching
# =============================================================================

# Key features to compare
key_features = [
    'TOP/CSC year', 'GFA  (m²)', 'AC Area Percentage (%)', 'Age of Chiller (year)', 
    'Occupancy Rate (%)', 'LED Percentage Usage (%)', 'Size Encoded', 'PV Encoded', 
    'Type of ACS_District Cooling Plant', 'Type of ACS_Others', 
    'Type of ACS_Water Cooled Chilled Water Plant'
]

# Revert to a non-standardized dataset with propensity scores
folder_path_nostd = r'08_Preparing for models'
file_name1_nostd = r'02 - Energy Performance Data from 2016 to 2021_for_IPW_out of blank_variables selected.xlsx'
std_keys = ['TOP/CSC year', 'GFA  (m²)', 'AC Area Percentage (%)', 
            'Age of Chiller (year)', 'Occupancy Rate (%)', 'LED Percentage Usage (%)']
file_path1_nostd = os.path.join(folder_path_nostd, file_name1_nostd)
df1_nostd = pd.read_excel(file_path1_nostd)
df_PScore_and_Indices_nostd = df1.copy()
# Overall
df_PScore_and_Indices_nostd[std_keys] = df1_nostd[std_keys]
# Treatment and control groups
treatment_group_nostd = df_PScore_and_Indices_nostd[df_PScore_and_Indices_nostd['Green Labeling'] == 1]
control_group_nostd = df_PScore_and_Indices_nostd[df_PScore_and_Indices_nostd['Green Labeling'] == 0]
# Virtual control and treatment groups
average_control_group_nostd = calculate_average_matched_group(treatment_group_nostd, df_PScore_and_Indices_nostd, neighbor_cols)
average_treatment_group_nostd = calculate_average_matched_group(control_group_nostd, df_PScore_and_Indices_nostd, neighbor_cols)

# Data output
file_name1_nostd = r'00 - Energy Performance Data from 2016 to 2021_Pscore_nostd.xlsx'
file_name2_nostd = r'01 - Energy Performance Data from 2016 to 2021_Pscore_treatment group_nostd.xlsx'
file_name3_nostd = r'02 - Energy Performance Data from 2016 to 2021_Pscore_average control group_nostd.xlsx'
file_name5_nostd = r'04 - Energy Performance Data from 2016 to 2021_Pscore_control group_nostd.xlsx'
file_name6_nostd = r'05 - Energy Performance Data from 2016 to 2021_Pscore_average treatment group_nostd.xlsx'
file_path1_nostd = os.path.join(folder_path, file_name1_nostd)
file_path2_nostd = os.path.join(folder_path, file_name2_nostd)
file_path3_nostd = os.path.join(folder_path, file_name3_nostd)
file_path5_nostd = os.path.join(folder_path, file_name5_nostd)
file_path6_nostd = os.path.join(folder_path, file_name6_nostd)
# Output overall data
df_PScore_and_Indices_nostd.to_excel(file_path1_nostd, index=False)
# Output treatment and control groups
treatment_group_nostd.to_excel(file_path2_nostd, index=False)
control_group_nostd.to_excel(file_path5_nostd, index=False)
# Output virtual control and treatment groups
average_control_group_nostd.to_excel(file_path3_nostd, index=False)
average_treatment_group_nostd.to_excel(file_path6_nostd, index=False)

# Descriptive statistics and visualization of feature distributions
# ====================
# View descriptive statistics
# Pre-matching - overall
df1_stats_pre = df_PScore_and_Indices_nostd[key_features].describe()  # Excluding EUI, these key features are used in logistic regression
df1_stats_pre_all = df_PScore_and_Indices_nostd.describe()  # Output overall description

# Post-matching - ATET - treatment group and virtual control group
treatment_group_post = treatment_group_nostd[key_features].describe()
treatment_group_post_all = treatment_group_nostd.describe()
average_control_group_post = average_control_group_nostd[key_features].describe()
average_control_group_post_all = average_control_group_nostd.describe()

# Post-matching - ATENT - control group and virtual treatment group
control_group_post = control_group_nostd[key_features].describe()
control_group_post_all = control_group_nostd.describe()
average_treatment_group_post = average_treatment_group_nostd[key_features].describe()
average_treatment_group_post_all = average_treatment_group_nostd.describe()

# Visualize feature distributions before and after matching and save the plots
for feature in key_features:
    plt.figure(figsize=(12, 6))

    # Plot the overall distribution before matching
    sns.kdeplot(df_PScore_and_Indices_nostd[feature], label='Pre-Match - Overall', color='blue', shade=True, common_norm=True, bw='silverman')
    # Plot the distribution of the treatment group after matching
    sns.kdeplot(treatment_group_nostd[feature], label='Post-Match - Treatment Group', color='green', shade=True, common_norm=True, bw='silverman')
    # Plot the distribution of the virtual control group after matching
    sns.kdeplot(average_control_group_nostd[feature], label='Post-Match - Average Control Group', color='red', shade=True, common_norm=True, bw='silverman')

    title = f'Distribution of {feature} - Before and After Matching (ATET)'
    # Replace special characters in the file name
    filename = title.replace('/', '_') + '.png'
    plt.xlabel(feature)
    plt.ylabel('Density')
    min_value = min(df_PScore_and_Indices_nostd[feature].min(),
                    treatment_group_nostd[feature].min(),
                    average_control_group_nostd[feature].min())

    max_value = max(df_PScore_and_Indices_nostd[feature].max(),
                    treatment_group_nostd[feature].max(),
                    average_control_group_nostd[feature].max())
    plt.xlim(min_value, max_value)
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'figs', filename), dpi=350)

    plt.figure(figsize=(12, 6))

    # Plot the overall distribution before matching
    sns.kdeplot(df_PScore_and_Indices_nostd[feature], label='Pre-Match - Overall', color='blue', shade=True, common_norm=True, bw='silverman')
    # Plot the distribution of the control group after matching
    sns.kdeplot(control_group_nostd[feature], label='Post-Match - Control Group', color='orange', shade=True, common_norm=True, bw='silverman')
    # Plot the distribution of the virtual treatment group after matching
    sns.kdeplot(average_treatment_group_nostd[feature], label='Post-Match - Average Treatment Group', color='purple', shade=True, common_norm=True, bw='silverman')

    title = f'Distribution of {feature} - Before and After Matching (ATENT)'
    # Replace special characters in the file name
    filename = title.replace('/', '_') + '.png'
    plt.xlabel(feature)
    plt.ylabel('Density')
    min_value = min(df_PScore_and_Indices_nostd[feature].min(),
                    control_group_nostd[feature].min(),
                    average_treatment_group_nostd[feature].min())

    max_value = max(df_PScore_and_Indices_nostd[feature].max(),
                    control_group_nostd[feature].max(),
                    average_treatment_group_nostd[feature].max())
    plt.xlim(min_value, max_value)
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'figs', filename), dpi=350)
# ====================
# Output comparison results
# The plots have been saved to the figs folder in the loop
# Output descriptive statistics
# Pre-matching - overall
df1_stats_pre.to_excel(os.path.join(folder_path, 'data_description', '00 - Description before matching-All.xlsx'))
df1_stats_pre_all.to_excel(os.path.join(folder_path, 'data_description', '00 - Description before matching-All_all.xlsx'))

# Post-matching - ATET - treatment group and virtual control group
treatment_group_post.to_excel(os.path.join(folder_path, 'data_description', '01 - Description after matching-Treatment group.xlsx'))
treatment_group_post_all.to_excel(os.path.join(folder_path, 'data_description', '01 - Description after matching-Treatment group_all.xlsx'))
average_control_group_post.to_excel(os.path.join(folder_path, 'data_description', '02 - Description after matching-Average control group.xlsx'))
average_control_group_post_all.to_excel(os.path.join(folder_path, 'data_description', '02 - Description after matching-Average control group_all.xlsx'))
# Post-matching - ATENT - control group and virtual treatment group
control_group_post.to_excel(os.path.join(folder_path, 'data_description', '03 - Description after matching-Control group.xlsx'))
control_group_post_all.to_excel(os.path.join(folder_path, 'data_description', '03 - Description after matching-Control group_all.xlsx'))
average_treatment_group_post.to_excel(os.path.join(folder_path, 'data_description', '04 - Description after matching-Average treatment group.xlsx'))
average_treatment_group_post_all.to_excel(os.path.join(folder_path, 'data_description', '04 - Description after matching-Average treatment group_all.xlsx'))  
# =============================================================================


# Part 4: Calculate ATET, ATENT, and ATE
# =============================================================================
# Reload the data
# Read overall data
df_PScore_and_Indices_nostd = pd.read_excel(file_path1_nostd)
# Read treatment and control groups
treatment_group_nostd = pd.read_excel(file_path2_nostd)
control_group_nostd = pd.read_excel(file_path5_nostd)
# Read virtual control and treatment groups
average_control_group_nostd = pd.read_excel(file_path3_nostd)
average_treatment_group_nostd = pd.read_excel(file_path6_nostd)

# EUI variables to calculate
eui_variables = [
    '2015 EUI (kWh/m².yr)', '2016 EUI (kWh/m².yr)', '2017 EUI (kWh/m².yr)',
    '2018 EUI (kWh/m².yr)', '2019 EUI (kWh/m².yr)', '2020 EUI (kWh/m².yr)',
    '2021 EUI (kWh/m².yr)'
]

# # Set up indices, as the real and virtual values do not correspond in the current index
# average_control_group_nostd_ordered = average_control_group_nostd.set_index(treatment_group_nostd.index)
# average_treatment_group_nostd_ordered = average_treatment_group_nostd.set_index(control_group_nostd.index)

# Estimate ATET - the difference between the treatment group EUI and the virtual control group EUI
for eui in eui_variables:
    atet_diff_var = eui + '_ATET_diff'
    treatment_group_nostd[atet_diff_var] = treatment_group_nostd[eui] - average_control_group_nostd[eui]

# Estimate ATENT - the difference between the control group EUI and the virtual treatment group EUI
# Note: The ATENT sign is virtual treatment group minus real control group
for eui in eui_variables:
    atent_diff_var = eui + '_ATENT_diff'
    control_group_nostd[atent_diff_var] = average_treatment_group_nostd[eui] - control_group_nostd[eui]

# Calculate the probability of treatment and control groups
p_treatment = df_PScore_and_Indices_nostd['Green Labeling'].mean()
p_control = 1 - p_treatment

# Calculate ATE, ATET, ATENT
ate_values = {}
atet_values = {}
atent_values = {}
for eui in eui_variables:
    atet_diff_var = eui + '_ATET_diff'
    atent_diff_var = eui + '_ATENT_diff'
    ate_values[eui + '_ATE'] = (treatment_group_nostd[atet_diff_var].mean() * p_treatment +
                                control_group_nostd[atent_diff_var].mean() * p_control)
    atet_values[atet_diff_var] = treatment_group_nostd[atet_diff_var].mean()
    atent_values[atent_diff_var] = control_group_nostd[atent_diff_var].mean()    
# =============================================================================

# Part 5: Result Testing
# =============================================================================
# Perform normality tests to check the distribution of ATET and ATENT
normality_test_results_ATET = {}
normality_test_results_ATENT = {}
for eui in eui_variables:
    # Test ATET
    atet_diff_var = eui + '_ATET_diff'
    stat, p_value = shapiro(treatment_group_nostd[atet_diff_var].dropna())
    normality_test_results_ATET[atet_diff_var + '_treatment'] = p_value
    # Test ATENT
    atent_diff_var = eui + '_ATENT_diff'
    stat, p_value = shapiro(control_group_nostd[atent_diff_var].dropna())
    normality_test_results_ATENT[atent_diff_var + '_control'] = p_value
    
# Output the normality test results of the original distribution, with very small p-values, indicating non-normal distribution
normality_test_results_ATET = pd.DataFrame(list(normality_test_results_ATET.items()), columns=['Variable', 'p-value'])   
normality_test_results_ATENT = pd.DataFrame(list(normality_test_results_ATENT.items()), columns=['Variable', 'p-value']) 
normality_test_results_ATET.to_excel(os.path.join(folder_path, 'data_description', '05 - Normality test results of ATET_before.xlsx'))
normality_test_results_ATENT.to_excel(os.path.join(folder_path, 'data_description', '06 - Normality test results of ATENT_before.xlsx'))

# Generate QQ plots and histograms for each EUI variable of the original treatment group and average control group, control group and average treatment group, ATET and ATENT, and store them
# ====================
# Define path
figs_path_for_nomality_test = os.path.join(folder_path, 'figs', 'Normality Test Before Strict Clean')
# Define QQ plot function
def generate_qqplot(data, feature_name):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    sm.graphics.qqplot(data.dropna(), line='s', ax=ax)
    plt.xlabel(feature_name)
    plt.ylabel('Quantiles')
    # Save the plot
    filename = feature_name.replace('/', '_').replace(':', '_') + '_QQ.png'
    plt.savefig(os.path.join(figs_path_for_nomality_test, filename), dpi=350)
    plt.close(fig)  # Close the figure to free memory
    
# Define histogram function
def plot_histogram(data, feature_name):
    plt.figure(figsize=(8, 6))
    plt.hist(data.dropna(), bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    # Save the plot
    filename = feature_name.replace('/', '_').replace(':', '_') + '_Histogram.png'
    plt.savefig(os.path.join(figs_path_for_nomality_test, filename), dpi=350)
    plt.close()  # Close the figure to free memory

for eui in eui_variables:
    # Treatment group and average control group
    plot_histogram(treatment_group_nostd[eui], f'Treatment Group {eui}')
    plot_histogram(average_control_group_nostd[eui], f'Average Control Group {eui}')
    generate_qqplot(treatment_group_nostd[eui], f'Treatment Group {eui}')
    generate_qqplot(average_control_group_nostd[eui], f'Average Control Group {eui}')
    # Control group and average treatment group
    plot_histogram(control_group_nostd[eui], f'Control Group {eui}')
    plot_histogram(average_treatment_group_nostd[eui], f'Average Treatment Group {eui}')
    generate_qqplot(control_group_nostd[eui], f'Control Group {eui}')
    generate_qqplot(average_treatment_group_nostd[eui], f'Average Treatment Group {eui}')
    # ATET and ATENT
    atet_diff_var = eui + '_ATET_diff'
    atent_diff_var = eui + '_ATENT_diff'
    plot_histogram(treatment_group_nostd[atet_diff_var], f'Treatment Group {atet_diff_var}')
    plot_histogram(control_group_nostd[atent_diff_var], f'Control Group {atent_diff_var}')
    generate_qqplot(treatment_group_nostd[atet_diff_var], f'Treatment Group {atet_diff_var}')
    generate_qqplot(control_group_nostd[atent_diff_var], f'Control Group {atent_diff_var}')
# ====================
    
# Remove outliers using the IQR method
# ====================
def remove_outliers_iqr(df, feature_name):
    Q1 = df[feature_name].quantile(0.25)
    Q3 = df[feature_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[feature_name] >= lower_bound) & (df[feature_name] <= upper_bound)]
    return df

# Create copies of treatment and average control groups, and control and average treatment groups
treatment_group_nostd_clean = treatment_group_nostd.copy()
control_group_nostd_clean = control_group_nostd.copy()

# EUI variables and lists of ATET, ATENT difference variables
eui_variables_all = eui_variables + [eui + '_ATET_diff' for eui in eui_variables] + [eui + '_ATENT_diff' for eui in eui_variables]
eui_variables_ATET = eui_variables + [eui + '_ATET_diff' for eui in eui_variables]
eui_variables_ATENT = eui_variables + [eui + '_ATENT_diff' for eui in eui_variables]

# Remove outliers from the treatment group
for feature in eui_variables_ATET:
    treatment_group_nostd_clean = remove_outliers_iqr(treatment_group_nostd_clean, feature)
# Use the cleaned treatment group index to filter the average control group (the indices were the same before this)
cleaned_indices_treatment_group_cleaned = treatment_group_nostd_clean.index
average_control_group_nostd_clean = average_control_group_nostd.loc[cleaned_indices_treatment_group_cleaned].copy()

# Remove outliers from the control group
for feature in eui_variables_ATENT:
    control_group_nostd_clean = remove_outliers_iqr(control_group_nostd_clean, feature)  
# Use the cleaned control group index to filter the average treatment group
cleaned_indices_control_group_cleaned = control_group_nostd_clean.index
average_treatment_group_nostd_clean = average_treatment_group_nostd.loc[cleaned_indices_control_group_cleaned].copy()

# Save the cleaned data
treatment_group_nostd_clean.to_excel(os.path.join(folder_path, '07 - Energy Performance Data from 2016 to 2021_treatment group all clean.xlsx'), index=False)
control_group_nostd_clean.to_excel(os.path.join(folder_path, '08 - Energy Performance Data from 2016 to 2021_control group all clean.xlsx'), index=False)
# ====================


# Check the distribution of histograms and QQ plots after processing
# ====================
figs_path_for_nomality_test = os.path.join(folder_path, 'figs', 'Normality Test After Strict Clean')
# Draw histograms for each EUI variable of the control group and average treatment group
for eui in eui_variables:
    # Treatment group and average control group
    plot_histogram(treatment_group_nostd_clean[eui], f'Treatment Group {eui}')
    generate_qqplot(treatment_group_nostd_clean[eui], f'Treatment Group {eui}')
    # Control group and average treatment group
    plot_histogram(control_group_nostd_clean[eui], f'Control Group {eui}')
    generate_qqplot(control_group_nostd_clean[eui], f'Control Group {eui}')
    # ATET and ATENT
    atet_diff_var = eui + '_ATET_diff'
    atent_diff_var = eui + '_ATENT_diff'
    plot_histogram(treatment_group_nostd_clean[atet_diff_var], f'Treatment Group {atet_diff_var}')
    plot_histogram(control_group_nostd_clean[atent_diff_var], f'Control Group {atent_diff_var}')
    generate_qqplot(treatment_group_nostd_clean[atet_diff_var], f'Treatment Group {atet_diff_var}')
    generate_qqplot(control_group_nostd_clean[atent_diff_var], f'Control Group {atent_diff_var}')
# ====================


# Re-estimate ATET, ATENT
# ====================
ate_values_after = {}
atet_values_after = {}
atent_values_after = {}
for eui in eui_variables:
    atet_diff_var = eui + '_ATET_diff'
    atent_diff_var = eui + '_ATENT_diff'
    ate_values_after[eui + '_ATE'] = (treatment_group_nostd_clean[atet_diff_var].mean() * p_treatment +
                                control_group_nostd_clean[atent_diff_var].mean() * p_control)
    atet_values_after[atet_diff_var] = treatment_group_nostd_clean[atet_diff_var].mean()
    atent_values_after[atent_diff_var] = control_group_nostd_clean[atent_diff_var].mean()
# ====================

# Re-perform normality tests
# ====================
normality_test_results_ATET_after = {}
normality_test_results_ATENT_after = {}
for eui in eui_variables:
    # Test ATET
    atet_diff_var = eui + '_ATET_diff'
    stat, p_value = shapiro(treatment_group_nostd_clean[atet_diff_var].dropna())
    normality_test_results_ATET_after[atet_diff_var + '_treatment'] = p_value
    # Test ATENT
    atent_diff_var = eui + '_ATENT_diff'
    stat, p_value = shapiro(control_group_nostd_clean[atent_diff_var].dropna())
    normality_test_results_ATENT_after[atent_diff_var + '_control'] = p_value
    
# Output the normality test results after cleaning, with very large p-values, indicating acceptance of the null hypothesis, i.e., normal distribution!
normality_test_results_ATET_after = pd.DataFrame(list(normality_test_results_ATET_after.items()), columns=['Variable', 'p-value'])   
normality_test_results_ATENT_after = pd.DataFrame(list(normality_test_results_ATENT_after.items()), columns=['Variable', 'p-value']) 
normality_test_results_ATET_after.to_excel(os.path.join(folder_path, 'data_description', '05 - Normality test results of ATET_after.xlsx'))
normality_test_results_ATENT_after.to_excel(os.path.join(folder_path, 'data_description', '06 - Normality test results of ATENT_after.xlsx'))    
# ====================
    
# Output ATE, ATET, ATENT values before and after processing
# ====================
ate_values = pd.DataFrame(list(ate_values.items()), columns=['Variable', 'ate'])
atet_values = pd.DataFrame(list(atet_values.items()), columns=['Variable', 'atet'])
atent_values = pd.DataFrame(list(atent_values.items()), columns=['Variable', 'atent'])    

ate_values_after = pd.DataFrame(list(ate_values_after.items()), columns=['Variable', 'ate'])
atet_values_after = pd.DataFrame(list(atet_values_after.items()), columns=['Variable', 'atet'])
atent_values_after = pd.DataFrame(list(atent_values_after.items()), columns=['Variable', 'atent'])  

ate_values.to_excel(os.path.join(folder_path, '09 - ATE before.xlsx'))
atet_values.to_excel(os.path.join(folder_path, '10 - ATET before.xlsx'))
atent_values.to_excel(os.path.join(folder_path, '11 - ATENT before.xlsx'))

ate_values_after.to_excel(os.path.join(folder_path, '09 - ATE_after.xlsx'))
atet_values_after.to_excel(os.path.join(folder_path, '10 - ATET_after.xlsx'))
atent_values_after.to_excel(os.path.join(folder_path, '11 - ATENT_after.xlsx'))
# ====================
    
    

# Check if cleaning only ATET and ATENT improves the remaining samples
# ====================
treatment_group_nostd_clean_ATET = treatment_group_nostd.copy()
control_group_nostd_clean_ATENT = control_group_nostd.copy()
eui_variables_ATET_only = [eui + '_ATET_diff' for eui in eui_variables]
eui_variables_ATENT_only = [eui + '_ATENT_diff' for eui in eui_variables]

# Remove outliers from the treatment group
for feature in eui_variables_ATET_only:
    treatment_group_nostd_clean_ATET = remove_outliers_iqr(treatment_group_nostd_clean_ATET, feature)
    
# Remove outliers from the control group
for feature in eui_variables_ATENT_only:
    control_group_nostd_clean_ATENT = remove_outliers_iqr(control_group_nostd_clean_ATENT, feature)

# PS: The results don't differ much, it seems that the EUI is the main driver, not ATET or ATENT

# Re-check if cleaning only EUI improves the remaining samples
treatment_group_nostd_clean_EUI = treatment_group_nostd.copy()
control_group_nostd_clean_EUI = control_group_nostd.copy()

# Remove outliers from the treatment group
for feature in eui_variables_ATET:
    treatment_group_nostd_clean_EUI = remove_outliers_iqr(treatment_group_nostd_clean_EUI, feature)

# Remove outliers from the control group
for feature in eui_variables_ATENT:
    control_group_nostd_clean_EUI = remove_outliers_iqr(control_group_nostd_clean_EUI, feature)

# PS: The results are the same as those obtained by cleaning both EUI and ATET/ATENT, indicating that cleaning EUI alone can achieve the most rigorous state

# Check if cleaning only for missing values improves the samples
# This ensures the samples are retained after removing missing values
# Define a function to remove missing values
def align_dataframes(df1, df2):
    # Remove rows with missing values
    df1_clean = df1.dropna()
    df2_clean = df2.dropna()
    # Ensure the indices are aligned
    common_indices = df1_clean.index.intersection(df2_clean.index)
    df1_aligned = df1_clean.loc[common_indices]
    df2_aligned = df2_clean.loc[common_indices]
    return df1_aligned, df2_aligned

# Samples after removing missing values
control_tight_sample_aligned, average_treatment_tight_aligned = align_dataframes(control_group_nostd, average_treatment_group_nostd)
treatment_tight_sample_aligned, average_control_tight_aligned = align_dataframes(treatment_group_nostd, average_control_group_nostd)
# Additional data output, added March 1, 2024
control_tight_sample_aligned.to_excel(os.path.join(folder_path, '04 - Energy Performance Data from 2016 to 2021_Pscore_control group_nostd_dropna.xlsx'), index=False)
average_treatment_tight_aligned.to_excel(os.path.join(folder_path, '05 - Energy Performance Data from 2016 to 2021_Pscore_average treatment group_nostd_dropna.xlsx'), index=False)
treatment_tight_sample_aligned.to_excel(os.path.join(folder_path, '01 - Energy Performance Data from 2016 to 2021_Pscore_treatment group_nostd_dropna.xlsx'), index=False)
average_control_tight_aligned.to_excel(os.path.join(folder_path, '02 - Energy Performance Data from 2016 to 2021_Pscore_average control group_nostd.xlsx'), index=False)

# Check the distribution of histograms and QQ plots after removing missing values
figs_path_for_nomality_test = os.path.join(folder_path, 'figs', 'Normality Test dropna')
# Draw histograms for each EUI variable of the control group and average treatment group
for eui in eui_variables:
    # Treatment group and average control group
    plot_histogram(treatment_tight_sample_aligned[eui], f'Treatment Group {eui}')
    plot_histogram(average_control_tight_aligned[eui], f'Average Control Group {eui}')
    generate_qqplot(treatment_tight_sample_aligned[eui], f'Treatment Group {eui}')
    generate_qqplot(average_control_tight_aligned[eui], f'Average Control Group {eui}')
    # Control group and average treatment group
    plot_histogram(control_tight_sample_aligned[eui], f'Control Group {eui}')
    plot_histogram(average_treatment_tight_aligned[eui], f'Average Treatment Group {eui}')
    generate_qqplot(control_tight_sample_aligned[eui], f'Control Group {eui}')
    generate_qqplot(average_treatment_tight_aligned[eui], f'Average Treatment Group {eui}')
    # ATET and ATENT
    atet_diff_var = eui + '_ATET_diff'
    atent_diff_var = eui + '_ATENT_diff'
    plot_histogram(treatment_tight_sample_aligned[atet_diff_var], f'Treatment Group {atet_diff_var}')
    plot_histogram(control_tight_sample_aligned[atent_diff_var], f'Control Group {atent_diff_var}')
    generate_qqplot(treatment_tight_sample_aligned[atet_diff_var], f'Treatment Group {atet_diff_var}')
    generate_qqplot(control_tight_sample_aligned[atent_diff_var], f'Control Group {atent_diff_var}')

# Check the normality test after removing missing values    
normality_test_results_ATET_dropna = {}
normality_test_results_ATENT_dropna = {}
for eui in eui_variables:
    # Test ATET
    atet_diff_var = eui + '_ATET_diff'
    stat, p_value = shapiro(treatment_tight_sample_aligned[atet_diff_var].dropna())
    normality_test_results_ATET_dropna[atet_diff_var + '_treatment'] = p_value
    # Test ATENT
    atent_diff_var = eui + '_ATENT_diff'
    stat, p_value = shapiro(control_tight_sample_aligned[atent_diff_var].dropna())
    normality_test_results_ATENT_dropna[atent_diff_var + '_control'] = p_value
    
# Output the normality test results after removing missing values, with very small p-values, indicating rejection of the null hypothesis, i.e., non-normal distribution
normality_test_results_ATET_dropna = pd.DataFrame(list(normality_test_results_ATET_dropna.items()), columns=['Variable', 'p-value'])   
normality_test_results_ATENT_dropna = pd.DataFrame(list(normality_test_results_ATENT_dropna.items()), columns=['Variable', 'p-value']) 
normality_test_results_ATET_dropna.to_excel(os.path.join(folder_path, 'data_description', '05 - Normality test results of ATET_dropna.xlsx'))
normality_test_results_ATENT_dropna.to_excel(os.path.join(folder_path, 'data_description', '06 - Normality test results of ATENT_dropna.xlsx'))   

# Output ATE, ATET, and ATENT at this stage    
ate_values_dropna = {}
atet_values_dropna = {}
atent_values_dropna = {}
for eui in eui_variables:
    atet_diff_var = eui + '_ATET_diff'
    atent_diff_var = eui + '_ATENT_diff'
    ate_values_dropna[eui + '_ATE'] = (treatment_tight_sample_aligned[atet_diff_var].mean() * p_treatment +
                                control_tight_sample_aligned[atent_diff_var].mean() * p_control)
    atet_values_dropna[atet_diff_var] = treatment_tight_sample_aligned[atet_diff_var].mean()
    atent_values_dropna[atent_diff_var] = control_tight_sample_aligned[atent_diff_var].mean()
    
ate_values_dropna = pd.DataFrame(list(ate_values_dropna.items()), columns=['Variable', 'ate'])
atet_values_dropna = pd.DataFrame(list(atet_values_dropna.items()), columns=['Variable', 'atet'])
atent_values_dropna = pd.DataFrame(list(atent_values_dropna.items()), columns=['Variable', 'atent'])    

ate_values_dropna.to_excel(os.path.join(folder_path, '09 - ATE dropna.xlsx'))
atet_values_dropna.to_excel(os.path.join(folder_path, '10 - ATET dropna.xlsx'))
atent_values_dropna.to_excel(os.path.join(folder_path, '11 - ATENT dropna.xlsx'))
# ====================



# Estimate p-values for ATET and ATENT, which can only be assessed after removing missing values, so p-values before processing cannot be provided
# ====================
# Estimate ATENT p-values, here we only process samples with missing values
atet_p_values_dropna = {}
atent_p_values_dropna = {}
    
for eui in eui_variables:
    atent_diff_var = eui + '_ATENT_diff'
    # Calculate ATENT
    atent_p_values_dropna[atent_diff_var] = ttest_rel(
        control_tight_sample_aligned[eui], average_treatment_tight_aligned[eui]
    ).pvalue

# Test ATET p-values
for eui in eui_variables:
    atet_diff_var = eui + '_ATET_diff'
    # Calculate ATET
    atet_p_values_dropna[atet_diff_var] = ttest_rel(
        treatment_tight_sample_aligned[eui], average_control_tight_aligned[eui]
    ).pvalue

# Ensure the samples after normality test
# Use paired samples t-test
atet_p_values_after = {}
atent_p_values_after = {}
# Use paired samples t-test
# Test ATET p-values
for eui in eui_variables:
    atet_diff_var = eui + '_ATET_diff'
    # You need to provide EUI values of the real control group and average control group
    # ttest_ind assumes independent samples, but here we have paired samples, so we need to use ttest_rel or Wilcoxon test
    # This code is just an example; in practice, you should choose the appropriate test based on the data
    atet_p_values_after[atet_diff_var] = ttest_rel(
        treatment_group_nostd_clean[eui], average_control_group_nostd_clean[eui]
    ).pvalue

# Test ATENT p-values
for eui in eui_variables:
    atent_diff_var = eui + '_ATENT_diff'
    # You need to provide EUI values of the real control group and average control group
    # ttest_ind assumes independent samples, but here we have paired samples, so we need to use ttest_rel or Wilcoxon test
    # This code is just an example; in practice, you should choose the appropriate test based on the data
    atent_p_values_after[atent_diff_var] = ttest_rel(
        control_group_nostd_clean[eui], average_treatment_group_nostd_clean[eui]
    ).pvalue
    
# Output p-test results before and after processing
# After removing missing values
atet_p_values_dropna = pd.DataFrame(list(atet_p_values_dropna.items()), columns=['Variable', 'atet p-values'])
atent_p_values_dropna = pd.DataFrame(list(atent_p_values_dropna.items()), columns=['Variable', 'atent p-values'])

atet_p_values_dropna.to_excel(os.path.join(folder_path, 'data_description', '07 - ATET p-values_dropna.xlsx'))
atent_p_values_dropna.to_excel(os.path.join(folder_path, 'data_description', '08 - ATENT p-values_dropna.xlsx'))
  
# After removing normal outliers
atet_p_values_after = pd.DataFrame(list(atet_p_values_after.items()), columns=['Variable', 'atet p-values'])
atent_p_values_after = pd.DataFrame(list(atent_p_values_after.items()), columns=['Variable', 'atent p-values'])

atet_p_values_after.to_excel(os.path.join(folder_path, 'data_description', '07 - ATET p-values_after.xlsx'))
atent_p_values_after.to_excel(os.path.join(folder_path, 'data_description', '08 - ATENT p-values_after.xlsx'))     
# ====================

# =============================================================================


# Part 6: Examine ATET and ATENT performance across different building types and green label types, added on March 8, 2024
# =============================================================================
treatment_group_nostd_clean = pd.read_excel(os.path.join(folder_path, '07 - Energy Performance Data from 2016 to 2021_treatment group all clean.xlsx'))
control_group_nostd_clean =  pd.read_excel(os.path.join(folder_path, '08 - Energy Performance Data from 2016 to 2021_control group all clean.xlsx'))
# Found that the 07 and 08 files already lost the green label type information, so it needs to be extracted and added from the original data; the file with the green label type is in folder 08, file 03
ori_df_path = r'08_Preparing for models\03 - Energy Performance Data from 2016 to 2021_for_IPW_OneHot1.xlsx'
ori_df = pd.read_excel(ori_df_path)

merged_treatment_df = pd.merge(treatment_group_nostd_clean, ori_df[['TOP/CSC year', 'GFA  (m²)', 'AC Area Percentage (%)', 'Age of Chiller (year)', 'Address', 'Green Mark Rating']],
                     on=['TOP/CSC year', 'GFA  (m²)', 'AC Area Percentage (%)', 'Age of Chiller (year)'],
                     how='inner')

# PS: Just realized that analyzing the control group is meaningless because the green label is definitely Nan
merged_control_df = pd.merge(control_group_nostd_clean, ori_df[['TOP/CSC year', 'GFA  (m²)', 'AC Area Percentage (%)', 'Age of Chiller (year)', 'Occupancy Rate (%)', 'LED Percentage Usage (%)', 'Address', 'Green Mark Rating']],
                     on=['TOP/CSC year', 'GFA  (m²)', 'AC Area Percentage (%)', 'Age of Chiller (year)', 'Occupancy Rate (%)', 'LED Percentage Usage (%)'],
                     how='inner')

# Calculate the average ATET under different green labels, added on March 8, 2024, to calculate the annual savings for each green label
# Select all ATET difference columns
atet_columns = [col for col in merged_treatment_df.columns if 'ATET_diff' in col]
# Calculate the average ATET under different Green Mark Ratings
average_atet_by_green_mark = merged_treatment_df.groupby('Green Mark Rating')[atet_columns].mean().mean(axis=1).reset_index()
# Rename columns for better readability
average_atet_by_green_mark.columns = ['Green Mark Rating', 'Average ATET']
# Calculate the frequency of each green label
green_mark_counts = merged_treatment_df['Green Mark Rating'].value_counts().reset_index()
green_mark_counts.columns = ['Green Mark Rating', 'Frequency']
# Calculate the total GFA
total_gfa_by_green_mark = merged_treatment_df.groupby('Green Mark Rating')['GFA  (m²)'].sum().reset_index()

merged_atet_by_green_mark = pd.merge(green_mark_counts, average_atet_by_green_mark, on='Green Mark Rating')
merged_atet_by_green_mark = pd.merge(merged_atet_by_green_mark, total_gfa_by_green_mark, on='Green Mark Rating')
merged_atet_by_green_mark.to_excel(os.path.join(folder_path, 'data_description', '09 - ATET by green mark type.xlsx'))
# =============================================================================

# Part 7: Just to redraw a nicer QQ plot, added on August 26, 2024
df_for_qqplot = pd.read_excel('07 - Energy Performance Data from 2016 to 2021_treatment group all clean.xlsx')

def plot_histogram2(data, feature_name, save_path=None, save=False, color='blue', tick_labelsize=12):
    plt.figure(figsize=(8, 6))
    plt.hist(data.dropna(), bins=30, edgecolor='#eeeeee', color=color)
    
    # Set tick label font and size
    plt.xticks(fontname='Times New Roman', fontsize=tick_labelsize)
    plt.yticks(fontname='Times New Roman', fontsize=tick_labelsize)
    locs, labels = plt.yticks()
    plt.yticks(locs, [int(label * 10) for label in locs])
    if save:
        filename = feature_name.replace('/', '_').replace(':', '_') + '_Histogram-2024年8月26日-final.png'
        plt.savefig(os.path.join(save_path, filename), dpi=350, bbox_inches='tight')
    
    plt.show()
    plt.close()
    

def generate_qqplot2(data, feature_name, save_path=None, save=False, scatter_color='#4d7191', line_color='red', scatter_size=100, tick_labelsize=12):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    qqplot = sm.ProbPlot(data.dropna())
    ax.scatter(qqplot.theoretical_quantiles, qqplot.sample_quantiles, color=scatter_color, s=scatter_size)
    qqplot.qqplot(line='s', ax=ax, color=line_color, alpha=0)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    
    if save:
        filename = feature_name.replace('/', '_').replace(':', '_') + '_QQ-2024年8月26日-final.png'
        plt.savefig(os.path.join(save_path, filename), dpi=350, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)
    
# selected_feature = '2020 EUI (kWh/m².yr)_ATET_diff'
selected_feature = '2020 EUI (kWh/m².yr)'

eui_columns = [f'{year} EUI (kWh/m².yr)' for year in range(2015, 2022)]
total_eui = pd.concat([df_for_qqplot[col] for col in eui_columns], ignore_index=True)


# Draw QQ plot and histogram
plot_histogram2(
    total_eui,
    f'Treatment Group {selected_feature}', 
    save_path=r'figs/Normality Test After Strict Clean',
    save=True,  # Whether to save
    color='#4d7191',
    tick_labelsize=22
)

generate_qqplot2(
    df_for_qqplot[selected_feature],
    f'Treatment Group {selected_feature}', 
    save_path=r'figs/Normality Test After Strict Clean',
    save=True,  # Whether to save
    scatter_color='#4d7191',  # Custom color
    line_color='red',
    scatter_size=50,
    tick_labelsize=22
)