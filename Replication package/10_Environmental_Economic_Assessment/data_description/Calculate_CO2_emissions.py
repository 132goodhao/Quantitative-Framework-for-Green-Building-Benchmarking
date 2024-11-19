# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:14:57 2024

@author: goodhao
"""

import pandas as pd

# Load the Excel file containing carbon reduction data
df = pd.read_excel('06 - CO reductions treatment_group_after.xlsx')
df.columns = df.columns.map(str)  # Ensure all column names are strings for consistency

# Initialize DataFrames
valid_df = pd.DataFrame(columns=df.columns)  # For storing valid carbon reduction data
result_df = pd.DataFrame()  # For storing aggregated statistical results

# Define building categories to aggregate
commercial = 'Commercial Building'
educational = 'Educational Institution'
result_df = pd.DataFrame(columns=['Year', 'Commercial Building', 'Educational Institution', 'Other Buildings'])

# Define the years of interest for analysis
years = ['2015', '2016', '2017', '2018', '2019', '2020', '2021']
result_df['Year'] = years
result_df[['Commercial Building', 'Educational Institution', 'Other Buildings']] = 0

# Process each row in the dataset
for index, row in df.iterrows():
    year_of_gma = int(row['Year of GMA'])  # Extract the Year of GMA (Green Mark Award)

    # Create a copy of the current row for validation
    valid_row = row.copy()

    # Clear invalid carbon reduction data for years before the Year of GMA
    for year in years:
        if int(year) < year_of_gma:
            valid_row[year] = 0

    # Append the validated row to the valid_df
    valid_df = pd.concat([valid_df, valid_row.to_frame().T], ignore_index=True)

    # Aggregate carbon reduction data by building type and year
    for year in years:
        if row['Type'] == 'Commercial Building':
            result_df.loc[result_df['Year'] == year, 'Commercial Building'] += valid_row[year]
        elif row['Type'] == 'Educational Institution':
            result_df.loc[result_df['Year'] == year, 'Educational Institution'] += valid_row[year]
        else:
            result_df.loc[result_df['Year'] == year, 'Other Buildings'] += valid_row[year]

# Save the validated data to a new Excel file
valid_df.to_excel('06 - CO reductions treatment_group_after_only_valid.xlsx', index=False)

# Save the aggregated statistical results to another Excel file
result_df.to_excel('07 - CO reductions treatment_group_after_statistical.xlsx', index=False)
