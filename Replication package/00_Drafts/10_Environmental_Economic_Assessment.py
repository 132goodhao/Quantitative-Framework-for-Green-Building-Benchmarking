# -*- coding: utf-8 -*-
"""
Environmental and Economic Benefit Assessment

@author: goodhao
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
from matplotlib.ticker import MaxNLocator
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportions_chisquare

# Calculate total energy savings in kWh, carbon emissions reductions in kgCO2, and economic benefits

# Part 1: Data Loading
# =============================================================================
# Files to be read
'''
Data sorted by year: ATE, ATET, ATENT, Emission Factors, Electricity Price
Data sorted by building samples: Year of GMA, Green Mark Rating, GFA
Constants: Occupancy Rate of treatment group: 87.75%, control group: 85.40%, all group: 86.66%, r_extra = 8.70%, 10.60%, 10.30%, 15.47%
'''
# Load data, using tight buildings, copied from folder 06 to folder 10
folder_path0 = r'10_Environmental_Economic_Assessment'
folder_path1 = r'10_Environmental_Economic_Assessment\data_description'
file_name1 = r'00 - Energy Performance Data from 2016 to 2021_tight_building.xlsx'
file_name2 = r'00 - Price and ATEs.xlsx'
file_path1 = os.path.join(folder_path0, file_name1)
file_path2 = os.path.join(folder_path0, file_name2)

df = pd.read_excel(file_path1)
df1 = pd.read_excel(file_path2)

# Keep only Address, GFA, Type
variable_GFA = ['Address', 'Type', 'GFA  (m²)', 'Year of GMA', 'Green Mark Rating']
df_EE_matrix = df[variable_GFA].copy()
df_price = df1
# =============================================================================

# Part 2: Calculate Energy Savings
# =============================================================================
df_EE_matrix_dropna = df_EE_matrix.copy()

def calculate_energy_savings(df, df_price, suffix):
    df_result = df.copy()
    for year in df_price['year']:
        column_name = f'E_{year}_{suffix} (kWh)'
        df_result[column_name] = 0

        for index, row in df_result.iterrows():
            effect = 'ATENT' if pd.isna(row['Year of GMA']) else 'ATET'
            saved_energy = row['GFA  (m²)'] * df_price.loc[df_price['year'] == year, f'{effect}_{suffix} (kWh/m²)'].values[0]
            df_result.at[index, column_name] = saved_energy
    return df_result

df_EE_matrix_dropna = calculate_energy_savings(df_EE_matrix, df_price, 'dropna')
df_EE_matrix_after = calculate_energy_savings(df_EE_matrix, df_price, 'after')

# Data output
out_file_name1 = r'01 - Treatment effect for energy savings for all buildings_dropna.xlsx'
out_file_name2 = r'01 - Treatment effect for energy savings for all buildings_after.xlsx'
df_EE_matrix_dropna.to_excel(os.path.join(folder_path0, out_file_name1), index=False)
df_EE_matrix_after.to_excel(os.path.join(folder_path0, out_file_name2), index=False)
# =============================================================================

# Part 3: Calculate CO2 Emissions Reductions and Present Value of Electricity Savings
# =============================================================================
def calculate_co2_and_electricity_savings(df, df_price, suffix, discount_rate=0.04):
    for year in range(2015, 2022):
        energy_saved_column = f'E_{year}_{suffix} (kWh)'
        co2_reductions_column = f'CO2_reductions_{year} (kg)'
        electricity_bill_reductions_column = f'Electricity_bill_reductions_{year} (S$)'
        discounted_savings_column = f'Discounted_Electricity_bill_{year}_to_2023 (S$)'

        emission_factor = df_price.loc[df_price['year'] == year, 'Emission Factors(kg CO2 / kWh)'].values[0]
        df[co2_reductions_column] = df[energy_saved_column] * emission_factor

        electricity_tariff = df_price.loc[df_price['year'] == year, 'Electricity Tariffs(S$/kWh)'].values[0]
        df[electricity_bill_reductions_column] = df[energy_saved_column] * electricity_tariff

        discount_factor = (1 + discount_rate) ** (2023 - year)
        df[discounted_savings_column] = df[electricity_bill_reductions_column] / discount_factor

    # Total
    co2_columns = [f'CO2_reductions_{year} (kg)' for year in range(2015, 2022)]
    df['Total_CO2_reductions (kg)'] = df[co2_columns].sum(axis=1)

    discounted_electricity_columns = [f'Discounted_Electricity_bill_{year}_to_2023 (S$)' for year in range(2015, 2022)]
    df['Total_Discounted_Electricity_Bill_Savings_to_2023 (S$)'] = df[discounted_electricity_columns].sum(axis=1)

    return df

# Use the function
df_EE_matrix_CO_PV_dropna = calculate_co2_and_electricity_savings(df_EE_matrix_dropna, df_price, 'dropna', 0.04)
df_EE_matrix_CO_PV_after = calculate_co2_and_electricity_savings(df_EE_matrix_after, df_price, 'after', 0.04)

# Data output
out_file_name3 = r'02 - CO and PV reductions for all buildings_dropna.xlsx'
out_file_name4 = r'02 - CO and PV reductions for energy savings for all buildings_after.xlsx'
df_EE_matrix_CO_PV_dropna.to_excel(os.path.join(folder_path0, out_file_name3), index=False)
df_EE_matrix_CO_PV_after.to_excel(os.path.join(folder_path0, out_file_name4), index=False)

# Calculate the actual effective benefits
# Split df_EE_matrix_dropna
df_EE_matrix_CO_PV_treatment_group_dropna = df_EE_matrix_CO_PV_dropna.dropna(subset=['Year of GMA'])
df_EE_matrix_CO_PV_control_group_dropna = df_EE_matrix_CO_PV_dropna[df_EE_matrix_dropna['Year of GMA'].isna()]

# Split df_EE_matrix_after
df_EE_matrix_CO_PV_treatment_group_after = df_EE_matrix_CO_PV_after.dropna(subset=['Year of GMA'])
df_EE_matrix_CO_PV_control_group_after = df_EE_matrix_CO_PV_after[df_EE_matrix_after['Year of GMA'].isna()]

# Calculate actual effective benefits after the Year of GMA, for the Treatment group
def calculate_co2_and_electricity_savings_post_gma(df, discount_rate=0.04):
    df = df.copy()  # Create a copy of df to avoid assignment risk on the original matrix
    co2_columns = [f'CO2_reductions_{year} (kg)' for year in range(2015, 2022)]
    discounted_electricity_columns = [f'Discounted_Electricity_bill_{year}_to_2023 (S$)' for year in range(2015, 2022)]

    # Add new columns
    df['Total_CO2_reductions_post_GMA (kg)'] = 0
    df['Total_Discounted_Electricity_Bill_Savings_post_GMA_to_2023 (S$)'] = 0

    for index, row in df.iterrows():
        gma_year = row['Year of GMA']
        # Sum only after the GMA year
        df.at[index, 'Total_CO2_reductions_post_GMA (kg)'] = row[[col for col in co2_columns if int(col.split('_')[2].split(' ')[0]) >= gma_year]].sum()
        df.at[index, 'Total_Discounted_Electricity_Bill_Savings_post_GMA_to_2023 (S$)'] = row[[col for col in discounted_electricity_columns if int(col.split('_')[3].split(' ')[0]) >= gma_year]].sum()

    return df

# Actual benefits generated from GMA implementation to present
df_EE_matrix_CO_PV_treatment_group_after = calculate_co2_and_electricity_savings_post_gma(df_EE_matrix_CO_PV_treatment_group_after)
df_EE_matrix_CO_PV_treatment_group_dropna = calculate_co2_and_electricity_savings_post_gma(df_EE_matrix_CO_PV_treatment_group_dropna)
# Slightly process by dropping the Year of GMA being 2022, as the data for 2022 is still not published as of December 2, 2023, but some YGMA is labeled as 2022, which seems to be a statistical error
df_EE_matrix_CO_PV_treatment_group_after = df_EE_matrix_CO_PV_treatment_group_after[df_EE_matrix_CO_PV_treatment_group_after['Year of GMA'] != 2022]
df_EE_matrix_CO_PV_treatment_group_dropna = df_EE_matrix_CO_PV_treatment_group_dropna[df_EE_matrix_CO_PV_treatment_group_dropna['Year of GMA'] != 2022]

# Data output
out_file_name5 = r'03 - CO and PV reductions for treatment_group_dropna.xlsx'
out_file_name6 = r'03 - CO and PV reductions for energy savings treatment_group_after.xlsx'
df_EE_matrix_CO_PV_treatment_group_after.to_excel(os.path.join(folder_path0, out_file_name6), index=False)
df_EE_matrix_CO_PV_treatment_group_dropna.to_excel(os.path.join(folder_path0, out_file_name5), index=False)
# =============================================================================

# Part 4: Visualization of Environmental and Economic Benefits, using "after" samples
# =============================================================================
# Reload data
df_EE_vis = df_EE_matrix_CO_PV_treatment_group_after.copy()
df_EE_vis = pd.read_excel(os.path.join(folder_path0, out_file_name6))

# Data preparation
# 1. Show the trend of CO2 reductions and changes in electricity savings over time
df_yearly_trends = pd.DataFrame({
    'Year': range(2015, 2022),
    'Total_CO2_Reductions': [df_EE_vis[df_EE_vis['Year of GMA'] <= year][f'CO2_reductions_{year} (kg)'].sum() for year in range(2015, 2022)],
    'Total_Discounted_Electricity_bill_Savings': [df_EE_vis[df_EE_vis['Year of GMA'] <= year][f'Discounted_Electricity_bill_{year}_to_2023 (S$)'].sum() for year in range(2015, 2022)]
})
df_yearly_trends = df_yearly_trends.abs()

# 2. Calculate statistics for different types of buildings using .describe()
df_building_type_stats = df_EE_vis.groupby('Type').agg({
    'Total_CO2_reductions_post_GMA (kg)': 'describe',
    'Total_Discounted_Electricity_Bill_Savings_post_GMA_to_2023 (S$)': 'describe'
}).reset_index()
df_building_type_stats.columns = ['_'.join(col).strip() if col[0] != 'Type' else col[0] for col in df_building_type_stats.columns.values]

# Sum totals
df_building_type_sums = df_EE_vis.groupby('Type').agg({
    'Total_CO2_reductions_post_GMA (kg)': 'sum',
    'Total_Discounted_Electricity_Bill_Savings_post_GMA_to_2023 (S$)': 'sum'
}).reset_index()
df_building_type_stats = df_building_type_stats.merge(df_building_type_sums, on='Type', suffixes=('', '_sum'))
df_building_type_stats['Type_Abbrev'] = ['CCC', 'CB', 'EI', 'HF', 'SRC']  # Set abbreviations for the x-axis
df_building_type_stats.iloc[:, 1:] = df_building_type_stats.iloc[:, 1:].abs()

# 3. The relationship between Year of GMA and reduction/saving effect
df_gma_relationship = df_EE_vis[['Year of GMA', 'Total_CO2_reductions_post_GMA (kg)', 'Total_Discounted_Electricity_Bill_Savings_post_GMA_to_2023 (S$)']]
# Group and sum the data by year
df_gma_sum = df_gma_relationship.groupby('Year of GMA').sum().reset_index()
df_gma_sum = df_gma_sum.abs()

df_gma_relationship = df_gma_relationship.abs()

# 4. CO2 reductions or present value of electricity savings by building type and year
# Pivot table - CO2 reductions
df_pivot_co2 = df_EE_vis.pivot_table(
    values='Total_CO2_reductions_post_GMA (kg)', 
    index=['Type'], 
    columns='Year of GMA', 
    aggfunc='sum'
)
Type_Abbrev = ['CCC', 'CB', 'EI', 'HF', 'SRC']
df_pivot_co2 = df_pivot_co2.reset_index()
df_pivot_co2.index = Type_Abbrev
df_pivot_co2.drop(columns='Type', inplace=True)
df_pivot_co2 = df_pivot_co2.abs()

# Pivot table - Electricity savings
df_pivot_electricity = df_EE_vis.pivot_table(
    values='Total_Discounted_Electricity_Bill_Savings_post_GMA_to_2023 (S$)', 
    index='Type', 
    columns='Year of GMA', 
    aggfunc='sum'
)
df_pivot_electricity = df_pivot_electricity.reset_index()
df_pivot_electricity.index = Type_Abbrev
df_pivot_electricity.drop(columns='Type', inplace=True)
df_pivot_electricity = df_pivot_electricity.abs()

# Data output
out_file_name7 = r'01 - CO and PV yearly trends reduction.xlsx'
out_file_name8 = r'02 - CO and PV describe according to building type.xlsx'
out_file_name9 = r'03 - CO and PV year of GMA and reduction.xlsx'
out_file_name10 = r'04 - CO yearly reduction matrix according to building type.xlsx'
out_file_name11 = r'05 - PV yearly reduction matrix according to building type.xlsx'

df_yearly_trends.to_excel(os.path.join(folder_path1, out_file_name7), index=False)
df_building_type_stats.to_excel(os.path.join(folder_path1, out_file_name8), index=False)
df_gma_relationship.to_excel(os.path.join(folder_path1, out_file_name9), index=False)
df_pivot_co2.to_excel(os.path.join(folder_path1, out_file_name10), index=False)
df_pivot_electricity.to_excel(os.path.join(folder_path1, out_file_name11), index=False)

# Data visualization
# 1. CO2 reductions and electricity savings trends over the years (trend chart)
# Create the chart and axes
dpi_value = 350
sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'  # Set font to Times New Roman
fig, ax1 = plt.subplots(figsize=(8, 6), dpi=dpi_value)
# Plot CO2 reductions trend
color1 = '#6B8520'
ax1.set_xlabel('Year')
ax1.set_ylabel('Total CO2 Reductions (kg)', color=color1)
ax1.plot(df_yearly_trends['Year'], df_yearly_trends['Total_CO2_Reductions'], color=color1, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xlim(2015, 2021)  # Set x-axis range

# Create the second y-axis for electricity savings
ax2 = ax1.twinx()  
color2 = '#BB4953'
ax2.set_ylabel('Total Electricity Bill Savings (S$)', color=color2) 
ax2.plot(df_yearly_trends['Year'], df_yearly_trends['Total_Discounted_Electricity_bill_Savings'], color=color2, linewidth=2, zorder=5)
ax2.tick_params(axis='y', labelcolor=color2)

# Adjust y-axis range
ax2.set_ylim(0, max(df_yearly_trends['Total_Discounted_Electricity_bill_Savings']) * 1.2)  # Adjust the upper limit
ax2.set_xlim(2015, 2021)  # Set x-axis range
ax2.grid(False)
# Adjust layout to avoid clipping
plt.tight_layout()
# Output the image
title = 'CO2 Reductions and Electricity Bill Savings Over Time'
filename = title + '.png'
plt.savefig(os.path.join(folder_path0, 'figs', filename), dpi=350)

# 2. CO2 reductions and electricity savings contribution by building type (pie charts)
dpi_value = 350
sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'  # Set font to Times New Roman
# CO2 reductions pie chart
plt.figure(figsize=(8, 8))
plt.pie(df_building_type_stats['Total_CO2_reductions_post_GMA (kg)'], labels=df_building_type_stats['Type_Abbrev'], autopct='%1.1f%%')
title = 'CO2 Reductions Contribution by Building Type'
filename = title + '.png'
plt.savefig(os.path.join(folder_path0, 'figs', filename), dpi=350)

# Electricity savings pie chart
plt.figure(figsize=(8, 8))
plt.pie(df_building_type_stats['Total_Discounted_Electricity_Bill_Savings_post_GMA_to_2023 (S$)'], labels=df_building_type_stats['Type_Abbrev'], autopct='%1.1f%%')
title = 'Electricity Bill Savings Contribution by Building Type'
filename = title + '.png'
plt.savefig(os.path.join(folder_path0, 'figs', filename), dpi=350)

# 3. The relationship between Year of GMA and reduction/saving effect
# CO2 reductions
dpi_value = 350
sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'  # Set font to Times New Roman
fig, ax1 = plt.subplots(figsize=(8, 6), dpi=dpi_value)
# Set bar width and offset
bar_width = 0.3
offset = 0.2

# CO2 reductions bar chart
color1 = '#6B8520'
ax1.set_xlabel('Year of GMA')
ax1.set_ylabel('Total CO2 Reductions (kg)', color=color1)
ax1.bar(df_gma_sum['Year of GMA'] - offset, df_gma_sum['Total_CO2_reductions_post_GMA (kg)'], color=color1, width=bar_width)
ax1.tick_params(axis='y', labelcolor=color1)

# Create a second axis sharing the x-axis
ax2 = ax1.twinx()
color2 = '#BB4953'
ax2.set_ylabel('Total Electricity Bill Savings (S$)', color=color2) 
ax2.bar(df_gma_sum['Year of GMA'] + offset, df_gma_sum['Total_Discounted_Electricity_Bill_Savings_post_GMA_to_2023 (S$)'], color=color2, width=bar_width)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.grid(False)

# Set x-axis range to ensure all bars are fully displayed
ax1.set_xlim(2014.5, 2021.5)

# Set chart title and display chart
fig.tight_layout()
title = 'CO2 Reductions and Electricity Bill Savings by Year of GMA'
filename = title + '.png'
plt.savefig(os.path.join(folder_path0, 'figs', filename), dpi=350)

# 4. CO2 reductions and electricity savings by building type and year (heatmap)
# Pivot table - CO2 reductions heatmap
dpi_value = 350
sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'  # Set font to Times New Roman
fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi_value)

sns.heatmap(df_pivot_co2[[year for year in df_pivot_co2.columns if 2015 <= year <= 2021]], cmap="YlGnBu", ax=ax)
ax.set_ylabel('Building Type')
ax.set_xlabel('Year of GMA')
title = 'CO2 Reductions Heatmap'
filename = title + '.png'
plt.savefig(os.path.join(folder_path0, 'figs', filename), dpi=350)

# Pivot table - Electricity savings heatmap
dpi_value = 350
sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'  # Set font to Times New Roman
fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi_value)
sns.heatmap(df_pivot_electricity[[year for year in df_pivot_electricity.columns if 2015 <= year <= 2021]], cmap="YlGnBu", ax=ax)
ax.set_ylabel('Building Type')
ax.set_xlabel('Year of GMA')
title = 'Electricity Bill Savings Heatmap'
filename = title + '.png'
plt.savefig(os.path.join(folder_path0, 'figs', filename), dpi=350)