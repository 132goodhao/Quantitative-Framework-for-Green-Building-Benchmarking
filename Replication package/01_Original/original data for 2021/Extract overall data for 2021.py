# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:07:59 2023
@author: goodhao
"""

import pandas as pd

# input
excel_file = pd.ExcelFile(r'01_Original\Listing of Building Energy Performance Data for 2021 (多个sheet).xlsx')

# get sheet name
sheet_names = excel_file.sheet_names

# get sheet 
sheet_name = 'All Buildings'

# output
df = excel_file.parse(sheet_name)
df.to_excel(r'01_Original\Listing of Building Energy Performance Data for 2021.xlsx', index=False)
