# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:00:28 2023

@author: goodhao
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.optimize import minimize

folder_path = r'05_Merged_and_unified'
file_name = r'Energy Performance Data from 2016 to 2021_unified_02.xlsx'
file_path = os.path.join(folder_path, file_name)

# 读取Excel文件
df = pd.read_excel(file_path)


# Part 1: 取整、排序、删除空值以及重置索引
# =============================================================================
# 四舍五入所有EUI到整数
EUI_variables = ['2015 EUI (kWh/m².yr)','2016 EUI (kWh/m².yr)','2017 EUI (kWh/m².yr)','2018 EUI (kWh/m².yr)',
                 '2019 EUI (kWh/m².yr)','2020 EUI (kWh/m².yr)', '2021 EUI (kWh/m².yr)']
df[EUI_variables] = df[EUI_variables].round() # 取整
# 按照address进行排序
df = df.sort_values(by='Address', ascending=False)
# 删除 "Address" 列为空值的样本
df = df.dropna(subset=['Address'])
# 重置索引
df = df.reset_index(drop=True)
# 使用 nunique() 方法统计不同的地址数量
unique_address_count = df['Address'].nunique()
# =============================================================================


# Part 2: 冲突的检查与合并
# =============================================================================
# 除EUI以外需要合并的变量
Other_variable_str = ['Building Name', 'Type', 'Size', 'Public Sector', 'Type of ACS', 'PV', 'Voluntary Disclosure']
Other_variable_num = ['TOP/CSC year', 'GFA  (m²)', 'AC Area (m²)', 'AC Area Percentage (%)']
# 创建一个用于存储冲突样本的列表
conflict_samples_for_EUI = [] # 存储EUI的冲突, 定义相同地址EUI之差大于2则冲突
conflict_samples_for_str = [] # 存储str变量的冲突，定义相同地址str不同则冲突
conflict_samples_for_num = [] # 存储除了EUI外的Num变量的冲突，定义相同地址绝对差大于20%则冲突

# 遍历每个唯一的 "Address"
unique_addresses = df['Address'].unique()

for address in unique_addresses:
    # 获取具有相同 "Address" 的样本
    address_samples = df[df['Address'] == address]
    
    # 01 EUI的冲突检查
    for variable in EUI_variables:
        # 获取该变量的非空值
        non_empty_values = address_samples[variable].dropna()
        
        # 如果没有非空值，跳过
        if non_empty_values.empty:
            continue
        # 计算非空值之间的差
        value_diff = non_empty_values.max() - non_empty_values.min()
       
        # 检查非空值是否有冲突
        if value_diff <= 2:
            # 无冲突，用最大值覆盖所有值
            df.loc[address_samples.index, variable] = non_empty_values.max()
        else:
            # 有冲突，将样本的索引添加到冲突列表
            conflict_samples_for_EUI.extend(address_samples.index)
        
    
    # 02 str变量的冲突检查
    for variable in Other_variable_str:
        values = address_samples[variable]
        non_empty_values = values.dropna()
        unique_values = non_empty_values.unique()
        
        if not non_empty_values.empty:
            # 统计非空值的频次
            value_counts = non_empty_values.value_counts()
            
            # 获取频次最高的值
            most_common_value = value_counts.idxmax()
            
            # 将频次最高的值覆盖所有其他值
            df.loc[address_samples.index, variable] = most_common_value
        
        # 检查是否有冲突
        if len(value_counts) > 1:
            # 如果有多个唯一值，存在冲突，将冲突的样本索引添加到冲突样本列表
            conflict_samples_for_str.extend(address_samples.index)
            
            
    # 03 其他num变量的冲突检查
    for variable in Other_variable_num:
        # 获取该变量的非空值
        non_empty_values = address_samples[variable].dropna()
        
        # 如果没有非空值，跳过
        if non_empty_values.empty:
            continue
        # 计算非空值之间的差
        max_value = non_empty_values.max()
        min_value = non_empty_values.min()
        range_threshold = (max_value - min_value) * 0.2
        # 检查非空值是否有冲突
        if (max_value - min_value) <= range_threshold:
            # 无冲突，用最大值覆盖所有值
            df.loc[address_samples.index, variable] = non_empty_values.max()
        else:
            # 有冲突，将样本的索引添加到冲突列表
            conflict_samples_for_num.extend(address_samples.index)           
        
# =============================================================================

# 输出文件
output_folder_path = r'06_Unified_and_Cleaned'
output_file_name = r'01 Energy Performance Data from 2016 to 2021_unified.xlsx'
output_file_path = os.path.join(output_folder_path, output_file_name)
df.to_excel(output_file_path, index=False)


# Part 3: tight文件输出及可视化
# =============================================================================
# 01 以建筑为标准，每栋建筑只有唯一一行数据
output_file_name_tight_building = r'02 Energy Performance Data from 2016 to 2021_tight_building.xlsx'
tight_building_path = os.path.join(output_folder_path, output_file_name_tight_building)
tight_building = []
unique_addresses = df['Address'].unique()
for address in unique_addresses:
    address_samples = df[df['Address'] == address]
    if not address_samples.empty:
        # 找到具有最少空值的样本
        min_empty_count = address_samples.isna().sum(axis=1).min()
        selected_sample = address_samples[address_samples.isna().sum(axis=1) == min_empty_count].iloc[0]
        # 将选定的样本添加到 tight_building DataFrame
        tight_building.append(selected_sample)
   
tight_building_df = pd.DataFrame(tight_building)
# 删除所有EUI都为空值的样本
tight_building_df = tight_building_df.dropna(subset=EUI_variables, how='all')
# 输出文件
tight_building_df.to_excel(tight_building_path, index=False)

# # 再读取一次
tight_building_df = pd.read_excel(tight_building_path)

# 可视化，yGMA的分布
yGMA_counts = (tight_building_df['Year of GMA']
               .value_counts()
               .reset_index() # 从Series转换成DataFrame方便后面排序
               .set_index('Year of GMA') # 主动设置一个index，避免增添默认的索引
               .sort_index(ascending=False))
# 输出yGMA的数据
yGMA_counts.to_excel(os.path.join(output_folder_path,'data', 'yGMA_counts.xlsx'))

plt.figure(dpi=300)
yGMA_counts.plot(kind='barh')
plt.xlabel('Numbers')
plt.ylabel('Year of GMA')
title = 'Year of GMA Distribution'
plt.title(title)
plt.savefig(os.path.join(output_folder_path, 'figs', title + '11.png'), bbox_inches='tight', dpi=750)

# 可视化，Green Mark Rating的分布
GMR_counts = (tight_building_df['Green Mark Rating']
               .value_counts()
               .reset_index()
               .set_index('Green Mark Rating')
               .reindex(['Platinum', 'GoldPlus', 'Gold', 'Certified', 'Legislated']))

plt.figure(dpi=300)
GMR_counts.plot(kind='barh')
plt.xlabel('Numbers')
plt.ylabel('Green Mark Rating')
title = 'Distribution of Green Mark Rating'
plt.title(title)
plt.savefig(os.path.join(output_folder_path, 'figs', title + '.png'), bbox_inches='tight', dpi=750)

# 可视化，Type of ACS的分布
TACS_counts = (tight_building_df['Type of ACS']
               .value_counts()
               .reset_index()
               .set_index('Type of ACS')
               .reindex(['Air Cooled Chilled Water Plant', 'Water Cooled Chilled Water Plant', 'Water Cooled Packaged Unit', 'District Cooling Plant', 'Others']))

plt.figure(dpi=300)
TACS_counts.plot(kind='barh')
plt.xlabel('Numbers')
plt.ylabel('Type of air conditioning systems')
title = 'Distribution of type of air conditioning systems'
plt.title(title)
plt.savefig(os.path.join(output_folder_path, 'figs', title + '.png'), bbox_inches='tight', dpi=750)

# 可视化，Type的分布
Type_counts = (tight_building_df['Type']
               .value_counts()
               .reset_index()
               .set_index('Type'))

plt.figure(dpi=300)
Type_counts.plot(kind='barh')
plt.xlabel('Numbers')
plt.ylabel('Building Type')
title = 'Distribution of building type'
plt.title(title)
plt.savefig(os.path.join(output_folder_path, 'figs', title + '.png'), bbox_inches='tight', dpi=750)

# 可视化，Function的分布
Function_counts_order = ['ITE/Polytechnic', 'Private School', 'University', 
          'Sports Complex', 'Recreation Club',
          'Clinic','Hospital/Nursing Home',
          'Civic Institution','Community Institution','Cultural Institution',
          'Mixed Development','Retail','Hotel','Office']
Function_counts_order_reserved = Function_counts_order[::-1] # 倒过来排以便符合上面的Type分布
Function_counts = (tight_building_df['Function']
               .value_counts()
               .reset_index()
               .set_index('Function')
               .reindex(Function_counts_order_reserved))

plt.figure(dpi=300)
Function_counts.plot(kind='barh')
plt.xlabel('Numbers')
plt.ylabel('Building Function')
title = 'Distribution of building Function'
plt.title(title)
plt.savefig(os.path.join(output_folder_path, 'figs', title + '.png'), bbox_inches='tight', dpi=750)

# 可视化，Public Sector的分布
Public_Sector_counts = (tight_building_df['Public Sector']
               .value_counts()
               .reset_index()
               .set_index('Public Sector'))

plt.figure(dpi=300)
Public_Sector_counts.plot(kind='barh')
plt.xlabel('Numbers')
plt.ylabel('Public Sector')
title = 'Distribution of Public Sector'
plt.title(title)
plt.savefig(os.path.join(output_folder_path, 'figs', title + '.png'), bbox_inches='tight', dpi=750)


# 可视化，TOP/CSC year的分布
TOP_CSC_counts = (tight_building_df['TOP/CSC year']
               .value_counts()
               .reset_index()
               .set_index('TOP/CSC year')
               .sort_index(ascending=False))

plt.figure(dpi=300)
TOP_CSC_counts.plot(marker='o', linestyle='-', markersize=4)
plt.xlabel('Numbers')
plt.ylabel('TOP/CSC year')
title = 'Distribution of TOP or CSC year'
plt.title(title)
plt.legend(loc='upper center')
plt.savefig(os.path.join(output_folder_path, 'figs', title + '.png'), bbox_inches='tight', dpi=750)

# 可视化，Size的分布
Size_counts = (tight_building_df['Size']
               .value_counts()
               .reset_index()
               .set_index('Size')
               .sort_index(ascending=False))

plt.figure(dpi=300)
Size_counts.plot(kind='barh')
plt.xlabel('Size')
plt.ylabel('Numbers')
title = 'Distribution of Size'
plt.title(title)
plt.legend(loc='upper right')
plt.savefig(os.path.join(output_folder_path, 'figs', title + '.png'), bbox_inches='tight', dpi=750)

# 统计变量对应的值和频率    
# 定义变量名称
variable_distri_names = ['Type', 'Function', 'Size', 'Public Sector', 'TOP/CSC year',
                  'Green Mark Rating', 'Year of GMA', 'GM Version', 'Type of ACS',
                  'Age of Chiller (year)', 'Date of Last Audit/Health Check', 'PV',
                  'Voluntary Disclosure']
# 定义输出文件名
filename_variable_distribution = os.path.join(output_folder_path, 'data', 'variable_distribution.xlsx')
# 定义统计变量频次的函数
def variable_distribution(df, variable_names):
    distributions = {}  # 用于存储每个变量的分布结果
    for variable_name in variable_distri_names:
        # print(variable_name)
        variable_counts = (tight_building_df[variable_name]
                          .value_counts()
                          .reset_index()
                          .set_axis([variable_name, 'Frequency'], axis=1)  # Rename the columns directly
                          .sort_values(by=variable_name, ascending=False)
                          .reset_index(drop=True))
                          # .rename(columns={'index': variable_name, variable_name: 'Frequency'})
                          # .set_index(variable_name)
                          # .sort_index(ascending=False))
        distributions[variable_name] = variable_counts
    return distributions

# 定义输出函数
def save_variable_distribution_to_excel(variable_distri, file_name):
    # 使用ExcelWriter来写入Excel文件
    with pd.ExcelWriter(file_name) as writer:
        for variable_name, counts in variable_distri.items():
            safe_sheet_name = variable_name.replace('/', '-')
            # 将每个DataFrame写入不同的sheet，sheet名为变量名
            counts.to_excel(writer, sheet_name=safe_sheet_name)

# 输出变量的频次字典
variable_distri = variable_distribution(tight_building_df, variable_distri_names)
# 输出变量频次统计的excel
save_variable_distribution_to_excel(variable_distri, filename_variable_distribution)

# 输出tight_building中按照green mark分类的描述性统计
df_tight_treatment = tight_building_df[tight_building_df['Year of GMA'].notna()]  
df_tight_control = tight_building_df[tight_building_df['Year of GMA'].isna()]
# 描述性统计
df_describe_tight_treatment = df_tight_treatment.describe()
df_describe_tight_control = df_tight_control.describe()
# 输出
df_describe_tight_treatment.to_excel(os.path.join(output_folder_path, 'data', 'Describe of treatment in tight buildings.xlsx'))
df_describe_tight_control.to_excel(os.path.join(output_folder_path, 'data', 'Describe of control in tight buildings.xlsx'))

# =============================================================================



# Part 4: 根据Year of GMA筛选各年累积的绿建信息，注：24年3月22日添加
# =============================================================================
tight_building_df = pd.read_excel(tight_building_path)

def process_GB_data(df, output_path):
    # 定义数值类型变量列，用于统计描述
    numeric_columns = [
        "GFA  (m²)", "AC Area (m²)", "AC Area Percentage (%)", 
        "Age of Chiller (year)", "ACS efficiency", "Occupancy Rate (%)", 
        "LED Percentage Usage (%)", "2015 EUI (kWh/m².yr)", "2016 EUI (kWh/m².yr)", 
        "2017 EUI (kWh/m².yr)", "2018 EUI (kWh/m².yr)", "2019 EUI (kWh/m².yr)", 
        "2020 EUI (kWh/m².yr)", "2021 EUI (kWh/m².yr)"
    ]
    # 使用xlsxwriter
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    
    for year in range(2006, 2023): 
        
        filtered_df = df[df['Year of GMA'] <= year]        
        numeric_df = filtered_df[numeric_columns]
        
        # 计算汇总统计数据：和、平均值、最大值、最小值、标准差
        summary_df = pd.DataFrame([
            numeric_df.sum(),
            numeric_df.mean(),
            numeric_df.max(),
            numeric_df.min(),
            numeric_df.std()
        ], index=['Sum', 'Mean', 'Max', 'Min', 'Std'])
        
        # 将结果写入对应的年份sheet
        sheet_name = f'GB_{str(year)[-2:]}'
        summary_df.to_excel(writer, sheet_name=sheet_name)
    
    # 输出excel文件
    writer.close()
    
# 定义输出文件路径和名字
output_path_GB_by_year = os.path.join(output_folder_path, 'data', 'Describe of green buildings by year.xlsx')
process_GB_data(tight_building_df,output_path_GB_by_year)

# 单独统计一下绿建的GFA
def process_GB_variable_stats(df, variable_name, output_path):
    stats_list = []

    for year in range(2006, 2023):
        # 筛选每年的数据
        filtered_df = df[df['Year of GMA'] <= year]
        # 获取指定变量列的统计数据
        variable_stats = filtered_df[variable_name].agg(['sum', 'mean', 'max', 'min', 'std'])
        # 将年份和统计数据添加到GB_Variable_Stats DataFrame中
        stats_list.append({
            'year': year,
            f'{variable_name}_sum': variable_stats['sum'],
            f'{variable_name}_mean': variable_stats['mean'],
            f'{variable_name}_max': variable_stats['max'],
            f'{variable_name}_min': variable_stats['min'],
            f'{variable_name}_std': variable_stats['std']
        })

    # 输出GB_Variable_Stats DataFrame到指定的Excel文件
    GB_Variable_Stats = pd.DataFrame(stats_list)
    GB_Variable_Stats.to_excel(output_path, index=False)
    
output_path_GB_GFA = os.path.join(output_folder_path, 'data', 'Green buildings GFA by year.xlsx')
process_GB_variable_stats(tight_building_df,'GFA  (m²)', output_path_GB_GFA)

# 单独根据建成年份统计一下建筑的总面积GFA
def process_CSC_variable_stats(df, variable_name, output_path):
    stats_list = []
    df = df.dropna(subset=['TOP/CSC year'])
    df['TOP/CSC year'] = df['TOP/CSC year'].astype(int)
    start_year = df['TOP/CSC year'].min()
    end_year = df['TOP/CSC year'].max()

    for year in range(start_year, end_year + 1):
        # 筛选每年的数据,但要注意某年没有新建建筑的情况
        filtered_df = df[df['TOP/CSC year'] <= year]
        if filtered_df.empty:
            continue
        # 获取指定变量列的统计数据
        variable_stats = filtered_df[variable_name].agg(['sum', 'mean', 'max', 'min', 'std'])
        # 将年份和统计数据添加到GB_Variable_Stats DataFrame中
        stats_list.append({
            'year': year,
            f'{variable_name}_sum': variable_stats['sum'],
            f'{variable_name}_mean': variable_stats['mean'],
            f'{variable_name}_max': variable_stats['max'],
            f'{variable_name}_min': variable_stats['min'],
            f'{variable_name}_std': variable_stats['std']
        })

    # 输出GB_Variable_Stats DataFrame到指定的Excel文件
    CSC_Variable_Stats = pd.DataFrame(stats_list)
    CSC_Variable_Stats.to_excel(output_path, index=False)
    
output_path_CSC_GFA = os.path.join(output_folder_path, 'data', 'All buildings GFA by year.xlsx')
process_CSC_variable_stats(tight_building_df,'GFA  (m²)', output_path_CSC_GFA)   
# =============================================================================


# Part 5: 进行一些预测，包括2023-2030_GFA_ALL,GFA_GB, GFA_NBG, 2006-2021,2023-2030_EUI，注：24年3月22日添加
# =============================================================================
output_folder_path = r'06_Unified_and_Cleaned'
file_name2 = r'06 - GFA_EUI_year.xlsx'
file_name3 = r'06 - GFA_EUI_year_predicted.xlsx'
file_path2 = os.path.join(output_folder_path,file_name2)
# 读取数据
df = pd.read_excel(file_path2, sheet_name='Predicted_EUI')

# GFA-ALL预测，逻辑增长函数
# ====================
# 使用逻辑增长函数来预测GFA_ALL
def logistic_growth(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

x_data = df['Year'][:17]  # 假设数据直到2022年
y_data = df['GFA-ALL'][:17]
# 参数拟合，使用curve fit
params, _ = curve_fit(logistic_growth, x_data, y_data, p0=[max(y_data), 1, 2000])
# 预测2023-2030_GFA_ALL,实际上需要2006-2021的数据来计算R²
x_predict = np.arange(2006, 2031)
y_predict_all = logistic_growth(x_predict, *params)
# 计算R²
y_true_all = df.loc[df['Year'].between(2006, 2021), 'GFA-ALL']
y_pred_all = y_predict_all[:16]
r_squared_all = r2_score(y_true_all, y_pred_all)

plt.figure(figsize=(8, 6), dpi=600)
plt.plot(x_data, y_data, 'bo', label='Actual GFA-ALL')
plt.plot(x_predict, y_predict_all, 'r--', label=f'Predicted GFA-ALL (R² = {r_squared_all:.2f})')
plt.xlabel('Year')
plt.ylabel('GFA-ALL (m²)')
plt.legend()
# plt.title('GFA-ALL Prediction')
plt.tight_layout()
plt.savefig(f'{output_folder_path}/figs/GFA-ALL_predicted.png')
# ====================

# GFA-GB预测，考虑80%约束，直接使用boltzman函数进行预测，上限就是GFA-ALL的80%
# ====================
# 定义Boltzman函数
def boltzmann(x, x0, T, A, B):
    return (A - B) / (1 + np.exp(-(x - x0) / T)) + B

x_data2 = df['Year'][:17]
y_data2 = df['GFA-GB'][:17]
initial_guess = [2016, 5, max(y_predict_all) * 0.8, min(y_data2)]  # 假设x0在中间年份，T为初始斜率，A和B为曲线上下限，上限0.8，下限实际最小值

popt, pcov = curve_fit(boltzmann, x_data2, y_data2, p0=initial_guess)

x_predict_gb = np.arange(2006, 2031)
y_predict_gb = boltzmann(x_predict_gb, *popt)
# 计算R²
y_true_gb = df.loc[df['Year'].between(2006, 2021), 'GFA-GB']
y_pred_gb = y_predict_gb[:16]
r_squared_gb = r2_score(y_true_gb, y_pred_gb)

plt.figure(figsize=(8, 6), dpi=600)
plt.plot(x_data2, y_data2, 'bo', label='Actual GFA-GB')
plt.plot(x_predict_gb, y_predict_gb, 'r--', label=f'Predicted GFA-GB (R² = {r_squared_gb:.2f})')
plt.xlabel('Year')
plt.ylabel('GFA-GB (m²)')
plt.legend()
# plt.title('Predicted GFA-GB Using Boltzmann Function')
plt.tight_layout()
plt.savefig(f'{output_folder_path}/figs/GFA-GB_predicted.png')

# 2024年8月28日添加，美化绘图
def plot_gfa_prediction(x_data, y_data, x_predict, y_predict, r_squared, save=False, output_folder_path=None, scatter_size=50, tick_labelsize=22, line_color='red', scatter_color='#4d7191'):
    plt.figure(figsize=(8, 6), dpi=350)    
    plt.plot(x_data, y_data / 10**6, 'o', color=scatter_color, markersize=scatter_size, label='Actual GFA')
    plt.plot(x_predict, y_predict / 10**6, '--', color=line_color, label=f'Predicted GFA (R² = {r_squared:.3f})')
    plt.xlabel('')
    plt.ylabel('GFA(km²)', fontname='Times New Roman', fontsize=25, labelpad=5)
    plt.xticks(fontname='Times New Roman', fontsize=tick_labelsize)
    plt.yticks(fontname='Times New Roman', fontsize=tick_labelsize)
    plt.gca().yaxis.tick_left()
    plt.legend(fontsize=16, loc='upper left')
    plt.tight_layout()
    if save:
        filename = os.path.join(output_folder_path, 'GFA_predicted.png')
        plt.savefig(filename)    
    plt.show()
    plt.close()

output_folder_path = r'origin draft'

plot_gfa_prediction(
    x_data2,
    y_data2,
    x_predict_gb,
    y_predict_gb,
    r_squared_gb,
    save=True,
    output_folder_path=output_folder_path,
    scatter_size=10,
    scatter_color='#4d7191'
)

# 新加坡building energy benchmarking report 2023的合理外推展示
data_EUI = {
    'Year': [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'EUI': [322, 312, 317, 319, 309, 301, 290, 285, 283, 265, 257, 255, 226, 225, 227]
}
df_EUI = pd.DataFrame(data_EUI)

from sklearn.linear_model import LinearRegression

def plot_eui_prediction(df, save=False, output_folder_path=None, scatter_size=50, tick_labelsize=22, line_color='red', scatter_color='#4d7191'):
    X = df['Year'].values.reshape(-1, 1)
    y = df['EUI'].values
    model = LinearRegression()
    model.fit(X, y)
    years_extended = np.arange(2006, 2031).reshape(-1, 1)
    predicted_eui = model.predict(years_extended)
    r_squared = r2_score(y, model.predict(X))
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Year'], df['EUI'], color=scatter_color, label='Actual EUI', s=scatter_size)
    plt.plot(years_extended, predicted_eui, '--', color=line_color, label=f'Predicted EUI (R² = {r_squared:.3f})')
    plt.xlabel('')
    plt.ylabel('EUI (kWh/m²)', fontname='Times New Roman', fontsize=25, labelpad=5)
    plt.xticks(fontname='Times New Roman', fontsize=tick_labelsize)
    plt.yticks(fontname='Times New Roman', fontsize=tick_labelsize)
    plt.ylim(0, 400)
    # plt.xlim(2006, 2030)
    plt.legend(fontsize=16, loc='best')
    plt.tight_layout()
    if save and output_folder_path is not None:
        filename = os.path.join(output_folder_path, 'EUI_Prediction.png')
        plt.savefig(filename, dpi=300)    
    plt.show()
    plt.close()
    result_df = pd.DataFrame({
        'Year': np.arange(2006, 2031),
        'Actual_EUI': np.where(np.isin(np.arange(2006, 2031), df['Year']), df['EUI'].tolist() + [np.nan] * (25 - len(df)), np.nan),
        'Predicted_EUI': predicted_eui
    })

    # 输出到06\data
    if save and output_folder_path is not None:
        output_file = os.path.join(output_folder_path, 'EUI_Prediction_Results.xlsx')
        result_df.to_excel(output_file, index=False)

output_folder_path2 = r'06_Unified_and_Cleaned\data'

plot_eui_prediction(
    df_EUI,
    save=True,
    output_folder_path=output_folder_path2,
    scatter_size=50,
    tick_labelsize=22,
    line_color='#f68657',
    scatter_color='#a3a1a1'
)


# ====================

# EUI的预测，也使用boltzman函数，不过人为设定上下限20%，而且使用反S型
# ====================
def predict_and_fill_boltzmann(df, column_name, x0, T, A, VB):
    # 使用指定的参数进行预测
    y_data = df[column_name].dropna()
    # A = max(y_data3) * 1.2
    # B = min(y_data3) * 0.8
    x_data = df['Year'][df[column_name].notna()]
    y_pred  = boltzmann(x_data, x0, T, B, A) # 仅预测EUI有真实值对应的区间的值，即2015-2021，用于计算R²
    r_squared = r2_score(y_data, y_pred)
    
    df[f'{column_name}_predicted'] = boltzmann(df['Year'], x0, T, B, A) # 预测所有区间的值，即2006-2030
    df[f'{column_name}_R²'] = r_squared  # 记录R²
    
    plt.figure(figsize=(8, 6), dpi=600)
    plt.plot(x_data, y_data, 'bo', label=f'Actual {column_name}')
    plt.plot(df['Year'], df[f'{column_name}_predicted'], 'r--', label=f'Predicted {column_name} (R² = {r_squared:.2f})')
    plt.xlabel('Year')
    plt.ylabel(f'{column_name} (kWh/m²)')
    plt.legend()
    plt.title(f'{column_name} Prediction using Boltzmann Function')
    plt.tight_layout()
    plt.savefig(f'{output_folder_path}/figs/{column_name}_predicted.png')
    plt.close()

T=3
x0 = 2018
eui_columns = ['GB-real-EUI', 'NGB-real-EUI', 'ATET-EUI', 'ATENT-EUI']
for column in eui_columns:
    # 假设的A和B，应根据实际情况调整
    A = df[column].max() * 1.2
    B = df[column].min() * 0.8
    predict_and_fill_boltzmann(df, column, x0, T, A, B)

df['GFA-GB_predicted'] = y_predict_gb
df['GFA-ALL_predicted'] = y_predict_all
df['GFA-ALL_R²'] = np.nan
df['GFA-GB_R²'] = np.nan
df.loc[df['Year'].between(2006, 2021), 'GFA-ALL_R²'] = r_squared_all
df.loc[df['Year'].between(2006, 2021), 'GFA-GB_R²'] = r_squared_gb
    
# 数据输出
df.to_excel(os.path.join(output_folder_path,file_name3), index=False)



    
    
    
    
    