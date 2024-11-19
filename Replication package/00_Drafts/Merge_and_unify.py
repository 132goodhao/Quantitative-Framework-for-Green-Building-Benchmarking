# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 10:21:47 2023

@author: goodhao
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folder_path = r'04_Reformed_and_Merged'
file_name = r'Energy Performance Data from 2016 to 2021.xlsx'
file_path = os.path.join(folder_path, file_name)

# 读取Excel文件
df = pd.read_excel(file_path)

def select_var2_from_varset (selected_var, selected_set, selected_var2, figfolder_path=None):
    # 根据某一变量var选择变量下面某种属性set的数据
    selected_data = df[df[selected_var] == selected_set]

    # 统计每种set的对应的另一变量var2的样本数
    set_counts = selected_data[selected_var2].value_counts()

    # 作图表示每种var2的样本数
    # set_counts.plot(kind='bar')
    plt.figure(dpi=300)
    set_counts.plot(kind='barh')
    plt.xlabel('Sample Count')
    plt.ylabel(selected_var2)
    title = selected_var2 + ' ' + 'Distribution for ' + selected_var
    plt.title(title)
    # 保存图片
    if figfolder_path:        
        figname = selected_var2 + ' ' + 'Distribution for ' + selected_var + ' in ' + selected_set + '.png'
        figsave_path = os.path.join(figfolder_path, figname)
        plt.savefig(figsave_path, bbox_inches='tight')
    plt.show()      
    return set_counts

# 定义要筛选的变量
selected_var = 'Type'
selected_set = 'Healthcare Facility'
selected_var2 = 'Function'
figfolder_path = r'05_Merged_and_unified\images before unified'
set_counts = select_var2_from_varset(selected_var, selected_set, selected_var2, figfolder_path)


# 选择要替换名称的值
replacements_Healthcare = {
    'Community Hospital/ Nursing Home': 'Hospital/Nursing Home',
    'Nursing Home': 'Hospital/Nursing Home',
    'Hospital': 'Hospital/Nursing Home',
    'General Hospital (Public)/ Specialist Centre (Public)': 'Hospital/Nursing Home',
    'Private Hospital': 'Hospital/Nursing Home',
    'Polyclinic/ Private Clinic': 'Clinic',
    'Specialist Clinic': 'Clinic',
    'Private Clinic': 'Clinic',
    'Polyclinic': 'Clinic'
}

selected_set2 = list(set_counts.index)
df2 = df.copy()
df2[selected_var2] = df2[selected_var2].replace(replacements_Healthcare)

# 再进行一轮替换
set_counts = select_var2_from_varset(selected_var, 'Educational Institution', selected_var2, figfolder_path)
replacements_Educational = {
    'ITE': 'ITE/ Polytechnic',
    'Polytechnic': 'ITE/ Polytechnic',
}
df2[selected_var2] = df2[selected_var2].replace(replacements_Educational)

# 先输出一份文件
df2.to_excel(r'05_Merged_and_unified\Energy Performance Data from 2016 to 2021_unified_01.xlsx')



# 读取文件继续操作
df2 = pd.read_excel(r'05_Merged_and_unified\Energy Performance Data from 2016 to 2021_unified_01.xlsx', index_col=0)
df2.columns = df2.columns.str.strip() # 删除变量末尾的空格
df2 = df2.rename_axis('index') # 指定索引名字
#将Type中的重复项合并
df2['Type'].replace({'Civic, Community & Cultural Institution': 'Civic, Community and Cultural Institution', 'Sports & Recreation Centre': 'Sports and Recreation Centre'}, inplace=True)
# 将Function中的重复项合并
replacements_Function = {
    'General Hospital/ Specialist Centre (Public)': 'Hospital/Nursing Home',
    'Healthcare Facility': 'Hospital/Nursing Home',
    'Office Building': 'Office',
    'Private College': 'Private School',
    'Private Hospital (Private)': 'Hospital/Nursing Home',
    'Retail Building': 'Retail',
    'Specialist Centre (Public)': 'Hospital/Nursing Home',
    'TCM Clinic': 'Clinic',
    'Community Hospital': 'Hospital/Nursing Home',
    'Univerisity': 'University',
    'Educational Institution': 'University'
}
df2['Function'].replace(replacements_Function, inplace=True)
# 将Size中的Omit替换为np.nan
df2['Size'].replace({'Omit': np.nan}, inplace=True)

# 替换Missing values为np.nan
missing_values = ['-','N','N/A','NA']
variables_01 = ['Building Name', 'Address', 'Type', 'Function', 'Size', 'TOP/CSC year', 'Green Mark Rating',
             'Year of GMA', 'GM Version', 'GFA  (m²)', 'Type of ACS', 'AC Area (m²)', 'AC Area Percentage',
             'Age of Chiller (year)', 'ACS efficiency', 'Date of Last Audit/Health Check', 'Occupancy Rate (%)',
             'LED Percentage Usage (%)', '2015 EUI (kWh/m².yr)', '2016 EUI (kWh/m².yr)', '2017 EUI (kWh/m².yr)',
             '2018 EUI (kWh/m².yr)', '2019 EUI (kWh/m².yr)', '2020 EUI (kWh/m².yr)', '2021 EUI (kWh/m².yr)']
df2[variables_01] = df2[variables_01].replace(missing_values, np.nan)

# 统一Public Sector的N和Y
df2['Public Sector'].replace({'N': 'No', 'Y': 'Yes', '-': np.nan}, inplace=True)
# 统一Green Mark Rating的Not Certified
df2['Green Mark Rating'].replace({'Not Certified': np.nan}, inplace=True)
# 统一Type of ACS的种类
df2['Type of ACS'].replace({'Others (Split Units, Unitary Systems)':'Others'}, inplace=True)
# 统一AC Area Percentage的单位为%，并重命名AC Area Percentage为AC Area Percentage (%)
df2.rename(columns={"AC Area Percentage": "AC Area Percentage (%)"}, inplace=True)
def process_percentage(x):
    if isinstance(x, (int, float)) and 0 < x <= 1:
        return round(x * 100)
    elif pd.isna(x):
        return np.nan
    else:
        return round(x)  
df2['AC Area Percentage (%)'] = df2['AC Area Percentage (%)'].apply(process_percentage)
# 统一Occupancy Rate (%)的单位为%
df2['Occupancy Rate (%)'] = df2['Occupancy Rate (%)'].apply(process_percentage)
# 统一LED Percentage Usage (%)
df2['LED Percentage Usage (%)'] = df2['LED Percentage Usage (%)'].apply(process_percentage)
# 统一PV 和 Voluntary Disclosure的N,Y
df2['Voluntary Disclosure'].replace({'N': 'No', 'Y': 'Yes', '-': np.nan}, inplace=True)
df2['PV'].replace({'N': 'No', 'Y': 'Yes', '-': np.nan}, inplace=True)
# 处理Age of Chiller的018与2021文件以年为单位的问题，转化成年
df2['Age of Chiller (year)'] = df2.apply(
    lambda row: 2018 - row['Age of Chiller (year)'] if (row['Voluntary Disclosure'] is not np.nan) and (1980 <= row['Age of Chiller (year)'] <= 2016) else
                 2021 - row['Age of Chiller (year)'] if (row['Voluntary Disclosure'] is np.nan) and (1999 <= row['Age of Chiller (year)'] <= 2020) else
                 row['Age of Chiller (year)'],
    axis=1
)
# 处理Date of Last Audit/Health Check的单位问题
# 定义基准日期
base_date = pd.Timestamp('1900-01-01')
# 使用 apply 方法逐行处理数据,pd.DateOffset(int(x) 不用减去1900年多出来的1天，因为本身这数据从记得时候开始就多了1天了，只要不是人工记录的，excel都会有这个问题
df2['Date of Last Audit/Health Check'] = df2['Date of Last Audit/Health Check'].apply(
    lambda x: (pd.to_datetime(base_date + pd.DateOffset(int(x))).year if x > 2025 else x) if not pd.isna(x) else x
)


# 输出一份文件02
df2.to_excel(r'05_Merged_and_unified\Energy Performance Data from 2016 to 2021_unified_02.xlsx', index=False)


# 读取文件继续操作
df2 = pd.read_excel(r'05_Merged_and_unified\Energy Performance Data from 2016 to 2021_unified_02.xlsx')







