# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:14:57 2024

@author: 14356
"""

import pandas as pd

# 读取Excel文件
df = pd.read_excel('06 - CO reductions treatment_group_after.xlsx')
df.columns = df.columns.map(str) # 统一变量名的类型为str

valid_df = pd.DataFrame(columns=df.columns) # 有效碳排放数据
result_df = pd.DataFrame() # 统计后的数据

# 定义要汇总的建筑类型
commercial = 'Commercial Building'
educational = 'Educational Institution'
result_df = pd.DataFrame(columns=['Year', 'Commercial Building', 'Educational Institution', 'Other Buildings'])
years = ['2015', '2016', '2017', '2018', '2019', '2020', '2021']
result_df['Year'] = years
result_df[['Commercial Building', 'Educational Institution', 'Other Buildings']] = 0

# 处理每一行数据
for index, row in df.iterrows():
    year_of_gma = int(row['Year of GMA'])
    
    valid_row = row.copy()
    
    # 根据Year of GMA清除无效碳排放数据
    for year in years:
        if int(year) < year_of_gma:
            valid_row[year] = 0
            
    valid_df = pd.concat([valid_df, valid_row.to_frame().T], ignore_index=True)

    # 根据建筑类型汇总有效数据
    for year in years:
        if row['Type'] == 'Commercial Building':
            result_df.loc[result_df['Year'] == year, 'Commercial Building'] += valid_row[year]
        elif row['Type'] == 'Educational Institution':
            result_df.loc[result_df['Year'] == year, 'Educational Institution'] += valid_row[year]
        else:
            result_df.loc[result_df['Year'] == year, 'Other Buildings'] += valid_row[year]

# 保存到Excel文件
valid_df.to_excel('06 - CO reductions treatment_group_after_only_valid.xlsx', index=False)
result_df.to_excel('07 - CO reductions treatment_group_after_statistical.xlsx', index=False)
