#!/usr/bin/env python
# coding: utf-8

# # Grades Prediction with Neural Network
# 
# ---
# 
# ### Essential Libraries
# 
# Let us begin by importing the essential Python Libraries.
# 
# > NumPy : Library for Numeric Computations in Python  
# > Pandas : Library for Data Acquisition and Preparation  
# > Matplotlib : Low-level library for Data Visualization  
# > Seaborn : Higher-level library for Data Visualization
# 

# In[1]:


# Basic Libraries
from tensorflow import keras
import tensorflow as tf
# import math
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt  # we only need pyplot
sb.set()  # set the default Seaborn style for graphics


# ---
# 
# ## Data Acquisition
# 
# Source: Singapore Sport School
# Dataset :
# 
# - P6 Screening Master Template (7 Dec 2016)\_Last Sendout.csv
# - P6 Screening Master Template (11 Dec 2017)\_Last Sendout.csv
# - Template for Data Extraction (DL Project) 2016 Sec4 cohort Sendout.csv
# - Template for Data Extraction (DL Project) 2017 Sec4 cohort Sendout.csv
# - Template for Data Extraction (DL Project) 2018 Sec4 cohort Sendout.csv
# - Template for Data Extraction (DL Project) 2019 Sec4 cohort Sendout.csv
# - Template for Data Extraction (DL Project) 2020 Sec4 cohort Sendout.csv
# - Template for Data Extraction (DL Project) 2021 Sec4 cohort Sendout.csv
# 
# The dataset is `~.csv`; hence we use the `read_csv` function from Pandas.  
# Immediately after importing, take a quick look at the data using the `head` function.
# 
# Note : `header` is an optional input parameter to the function `read_csv`.
# 
# - If you do not input a `header` value, it will default to `infer`, taking (generally) the first row of the CSV file as column names.
# - If you set `header = None` then it will understand that there is no column names in the CSV file, and every row contains just data.
# - If you set `header = 0`, it will understand that you want the 0-th row (first row) of the CSV file to be considered as column names.
# 
# Check any function definition in Jupyter Notebook by running `function_name?`, e.g., try running `pd.read_csv?` in a cell.
# 

# In[2]:


import pandas as pd
# Error in reading file #
data_P6_2016 = pd.read_csv(
    'Research Data Sendout/P6 Screening Master Template (7 Dec 2016)_Last Sendout.csv', skiprows=1)
data_P6_2017 = pd.read_csv(
    'Research Data Sendout/P6 Screening Master Template (11 Dec 2017)_Last Sendout.csv', skiprows=1)
# Working Fine #
data_S4_2016 = pd.read_csv(
    'Research Data Sendout/Template for Data Extraction (DL Project) 2016 Sec4 cohort  Sendout.csv')
data_S4_2017 = pd.read_csv(
    'Research Data Sendout/Template for Data Extraction (DL Project) 2017 Sec4 cohort Sendout.csv')
data_S4_2018 = pd.read_csv(
    'Research Data Sendout/Template for Data Extraction (DL Project) 2018 Sec4 cohort Sendout.csv')
data_S4_2019 = pd.read_csv(
    'Research Data Sendout/Template for Data Extraction (DL Project) 2019 Sec4 cohort Sendout.csv')
data_S4_2020 = pd.read_csv(
    'Research Data Sendout/Template for Data Extraction (DL Project) 2020 Sec4 cohort Sendout.csv')
data_S4_2021 = pd.read_csv(
    'Research Data Sendout/Template for Data Extraction (DL Project) 2021 Sec4 cohort Sendout.csv')


# Dimensions of the dataset using `shape`.
# 

# In[3]:


print("data_P6_2016: ", data_P6_2016.shape)
print("data_P6_2017: ", data_P6_2017.shape)
print("=========================")
print("data_S4_2016: ", data_S4_2016.shape)
print("data_S4_2017: ", data_S4_2017.shape)
print("data_S4_2018: ", data_S4_2018.shape)
print("data_S4_2019: ", data_S4_2019.shape)
print("data_S4_2020: ", data_S4_2020.shape)
print("data_S4_2021: ", data_S4_2021.shape)


# Information about the dataset using `info()`.
# 

# In[4]:


data_S4_2016.info()


# `describe()` provides you with statistical information about the data. This is another method.
# 

# In[5]:


data_S4_2016.describe()


# In[6]:


print(list(data_S4_2016))


# In[7]:


print(list(data_S4_2017))


# In[8]:


print(list(data_S4_2018))


# In[9]:


print(list(data_S4_2019))


# In[10]:


print(list(data_S4_2020))


# In[11]:


print(list(data_S4_2021))


# # Data Cleaning
# 
# ---
# 

# In[12]:


# Truncating dataframe to columns that we only need
df_S4_2016 = pd.DataFrame(data_S4_2016[['Serial number', 'Gender', 'Race', 'Academy', 'Sec4_CardingLevel', 'Course',
                                        'TScore', 'MT', 'HMT', 'ResidentialType', 'Chinese', 'Chinese.1', 'Chinese.2', 'Chinese.3',
                                        'Higher Chinese', 'Higher Chinese.1', 'Higher Chinese.2', 'Higher Chinese.3',
                                        'Malay', 'Malay.1', 'Malay.2', 'Malay.3', 'Higher Malay', 'Higher Malay.1', 'Higher Malay.2', 'Higher Malay.3',
                                        'Hindi', 'Hindi.1', 'Hindi.2', 'Hindi.3']])
df_S4_2017 = pd.DataFrame(data_S4_2017[['Serial number', 'Gender', 'Race', 'Academy', 'Sec4_CardingLevel', 'Course',
                                        'TScore', 'MT', 'HMT', 'ResidentialType', 'Chinese', 'Chinese.1', 'Chinese.2', 'Chinese.3',
                                        'Higher Chinese', 'Higher Chinese.1', 'Higher Chinese.2', 'Higher Chinese.3',
                                        'Malay', 'Malay.1', 'Malay.2', 'Malay.3', 'Higher Malay', 'Higher Malay.1', 'Higher Malay.2', 'Higher Malay.3',
                                        'Tamil', 'Tamil.1', 'Tamil.2', 'Tamil.3']])
df_S4_2018 = pd.DataFrame(data_S4_2018[['Serial number', 'Gender', 'Race', 'Academy', 'Sec4_CardingLevel', 'Course',
                                        'TScore', 'MT', 'HMT', 'ResidentialType', 'Chinese', 'Chinese.1', 'Chinese.2', 'Chinese.3',
                                        'Higher Chinese', 'Higher Chinese.1', 'Higher Chinese.2', 'Higher Chinese.3',
                                        'Malay', 'Malay.1', 'Malay.2', 'Malay.3', 'Higher Malay', 'Higher Malay.1',
                                        'Tamil', 'Tamil.1', 'Tamil.2']])
df_S4_2019 = pd.DataFrame(data_S4_2019[['Serial number', 'Gender', 'Race', 'Academy', 'Sec4_CardingLevel', 'Course',
                                        'TScore', 'MT', 'HMT', 'ResidentialType', 'Chinese', 'Chinese.1', 'Chinese.2', 'Chinese.3',
                                        'Higher Chinese', 'Higher Chinese.1', 'Higher Chinese.2', 'Higher Chinese.3',
                                        'Malay', 'Malay.1', 'Malay.2', 'Malay.3',
                                        'Tamil', 'Tamil.1', 'Tamil.2', 'Tamil.3',
                                        "Chinese (GCE 'O' Level)", "Malay (GCE 'O' Level)", "Tamil (GCE 'O' Level)"]])
df_S4_2020 = pd.DataFrame(data_S4_2020[['Serial number', 'Gender', 'Race', 'Academy', 'Sec4_CardingLevel', 'Course',
                                        'TScore', 'MT', 'HMT', 'ResidentialType', 'Chinese', 'Chinese.1', 'Chinese.2', 'Chinese.3',
                                        'Higher Chinese', 'Higher Chinese.1', 'Higher Chinese.2', 'Higher Chinese.3',
                                        'Malay', 'Malay.1', 'Malay.2', 'Malay.3',
                                        'Tamil', 'Tamil.1', 'Tamil.2', 'Tamil.3',
                                        "Chinese (GCE 'O' Level)", "Malay (GCE 'O' Level)", "Tamil (GCE 'O' Level)"]])
df_S4_2021 = pd.DataFrame(data_S4_2021[['Serial number', 'Gender', 'Race', 'Academy', 'Sec4_CardingLevel', 'Course',
                                        'TScore', 'MT', 'HMT', 'ResidentialType', 'Chinese', 'Chinese.1', 'Chinese.2', 'Chinese.3',
                                        'Higher Chinese', 'Higher Chinese.1', 'Higher Chinese.2', 'Higher Chinese.3',
                                        'Malay', 'Malay.1', 'Malay.2', 'Malay.3',
                                        'Tamil', 'Tamil.1', 'Tamil.2',
                                        "Chinese (GCE 'O' Level)", "Malay (GCE 'O' Level)", "Tamil (GCE 'O' Level)"]])

# Renaming to avoid confusion
df_S4_2016 = df_S4_2016.rename(columns={
    'Higher Chinese': 'HCL_S4', 'Higher Chinese.1': 'HCL_S3', 'Higher Chinese.2': 'HCL_S2', 'Higher Chinese.3': 'HCL_S1',
                                'Chinese': 'CL_S4', 'Chinese.1': 'CL_S3', 'Chinese.2': 'CL_S2', 'Chinese.3': 'CL_S1',
                                'Higher Malay': 'HML_S4', 'Higher Malay.1': 'HML_S3', 'Higher Malay.2': 'HML_S2', 'Higher Malay.3': 'HML_S1',
                                'Malay': 'ML_S4', 'Malay.1': 'ML_S3', 'Malay.2': 'ML_S2', 'Malay.3': 'ML_S1',
                                'Hindi': 'TL_S4', 'Hindi.1': 'TL_S3', 'Hindi.2': 'TL_S2', 'Hindi.3': 'TL_S1'})
df_S4_2017 = df_S4_2017.rename(columns={
    'Higher Chinese': 'HCL_S4', 'Higher Chinese.1': 'HCL_S3', 'Higher Chinese.2': 'HCL_S2', 'Higher Chinese.3': 'HCL_S1',
                                'Chinese': 'CL_S4', 'Chinese.1': 'CL_S3', 'Chinese.2': 'CL_S2', 'Chinese.3': 'CL_S1',
                                'Higher Malay': 'HML_S4', 'Higher Malay.1': 'HML_S3', 'Higher Malay.2': 'HML_S2', 'Higher Malay.3': 'HML_S1',
                                'Malay': 'ML_S4', 'Malay.1': 'ML_S3', 'Malay.2': 'ML_S2', 'Malay.3': 'ML_S1',
                                'Tamil': 'TL_S4', 'Tamil.1': 'TL_S3', 'Tamil.2': 'TL_S2', 'Tamil.3': 'TL_S1'})
df_S4_2018 = df_S4_2018.rename(columns={
    'Higher Chinese': 'HCL_S4', 'Higher Chinese.1': 'HCL_S3', 'Higher Chinese.2': 'HCL_S2', 'Higher Chinese.3': 'HCL_S1',
                                'Chinese': 'CL_S4', 'Chinese.1': 'CL_S3', 'Chinese.2': 'CL_S2', 'Chinese.3': 'CL_S1',
                                'Higher Malay': 'HML_S2', 'Higher Malay.1': 'HML_S1',
                                'Malay': 'ML_S4', 'Malay.1': 'ML_S3', 'Malay.2': 'ML_S2', 'Malay.3': 'ML_S1',
                                'Tamil': 'TL_S3', 'Tamil.1': 'TL_S2', 'Tamil.2': 'TL_S1'})
df_S4_2019 = df_S4_2019.rename(columns={
    'Higher Chinese': 'HCL_S4', 'Higher Chinese.1': 'HCL_S3', 'Higher Chinese.2': 'HCL_S2', 'Higher Chinese.3': 'HCL_S1',
                                'Chinese': 'CL_S4', 'Chinese.1': 'CL_S3', 'Chinese.2': 'CL_S2', 'Chinese.3': 'CL_S1',
                                'Malay': 'ML_S4', 'Malay.1': 'ML_S3', 'Malay.2': 'ML_S2', 'Malay.3': 'ML_S1',
                                'Tamil': 'TL_S4', 'Tamil.1': 'TL_S3', 'Tamil.2': 'TL_S2', 'Tamil.3': 'TL_S1',
                                "Chinese (GCE 'O' Level)": 'CL_O', "Malay (GCE 'O' Level)": 'ML_O', "Tamil (GCE 'O' Level)": 'TL_O'})
df_S4_2020 = df_S4_2020.rename(columns={
    'Higher Chinese': 'HCL_S4', 'Higher Chinese.1': 'HCL_S3', 'Higher Chinese.2': 'HCL_S2', 'Higher Chinese.3': 'HCL_S1',
                                'Chinese': 'CL_S4', 'Chinese.1': 'CL_S3', 'Chinese.2': 'CL_S2', 'Chinese.3': 'CL_S1',
                                'Malay': 'ML_S4', 'Malay.1': 'ML_S3', 'Malay.2': 'ML_S2', 'Malay.3': 'ML_S1',
                                'Tamil': 'TL_S4', 'Tamil.1': 'TL_S3', 'Tamil.2': 'TL_S2', 'Tamil.3': 'TL_S1',
                                "Chinese (GCE 'O' Level)": 'CL_O', "Malay (GCE 'O' Level)": 'ML_O', "Tamil (GCE 'O' Level)": 'TL_O'})
df_S4_2021 = df_S4_2021.rename(columns={
    'Higher Chinese': 'HCL_S4', 'Higher Chinese.1': 'HCL_S3', 'Higher Chinese.2': 'HCL_S2', 'Higher Chinese.3': 'HCL_S1',
                                'Chinese': 'CL_S4', 'Chinese.1': 'CL_S3', 'Chinese.2': 'CL_S2', 'Chinese.3': 'CL_S1',
                                'Malay': 'ML_S4', 'Malay.1': 'ML_S3', 'Malay.2': 'ML_S2', 'Malay.3': 'ML_S1',
                                'Tamil': 'TL_S3', 'Tamil.1': 'TL_S2', 'Tamil.2': 'TL_S1',
                                "Chinese (GCE 'O' Level)": 'CL_O', "Malay (GCE 'O' Level)": 'ML_O', "Tamil (GCE 'O' Level)": 'TL_O'})


# In[13]:


# insert dummy columns
df_S4_2018.insert(df_S4_2018.columns.get_loc('HML_S2'), 'HML_S4', float('NaN'))
df_S4_2018.insert(df_S4_2018.columns.get_loc('HML_S2'), 'HML_S3', float('NaN'))
df_S4_2018.insert(df_S4_2018.columns.get_loc('TL_S3'), 'TL_S4', float('NaN'))

df_S4_2019.insert(df_S4_2019.columns.get_loc('TL_S4'), 'HML_S4', float('NaN'))
df_S4_2019.insert(df_S4_2019.columns.get_loc('TL_S4'), 'HML_S3', float('NaN'))
df_S4_2019.insert(df_S4_2019.columns.get_loc('TL_S4'), 'HML_S2', float('NaN'))
df_S4_2019.insert(df_S4_2019.columns.get_loc('TL_S4'), 'HML_S1', float('NaN'))

df_S4_2020.insert(df_S4_2020.columns.get_loc('TL_S4'), 'HML_S4', float('NaN'))
df_S4_2020.insert(df_S4_2020.columns.get_loc('TL_S4'), 'HML_S3', float('NaN'))
df_S4_2020.insert(df_S4_2020.columns.get_loc('TL_S4'), 'HML_S2', float('NaN'))
df_S4_2020.insert(df_S4_2020.columns.get_loc('TL_S4'), 'HML_S1', float('NaN'))

df_S4_2021.insert(df_S4_2021.columns.get_loc('TL_S3'), 'HML_S4', float('NaN'))
df_S4_2021.insert(df_S4_2021.columns.get_loc('TL_S3'), 'HML_S3', float('NaN'))
df_S4_2021.insert(df_S4_2021.columns.get_loc('TL_S3'), 'HML_S2', float('NaN'))
df_S4_2021.insert(df_S4_2021.columns.get_loc('TL_S3'), 'HML_S1', float('NaN'))
df_S4_2021.insert(df_S4_2021.columns.get_loc('TL_S3'), 'TL_S4', float('NaN'))


# In[14]:


print(df_S4_2016.shape)
print(df_S4_2017.shape)
print(df_S4_2018.shape)
print(df_S4_2019.shape)
print(df_S4_2020.shape)
print(df_S4_2021.shape)


# In[15]:


cols = ['HCL_S4', 'HCL_S3', 'HCL_S2', 'HCL_S1',
        'CL_S4', 'CL_S3', 'CL_S2', 'CL_S1',
        'HML_S4', 'HML_S3', 'HML_S2', 'HML_S1',
        'ML_S4', 'ML_S3', 'ML_S2', 'ML_S1',
        'TL_S4', 'TL_S3', 'TL_S2', 'TL_S1']


# In[16]:


# cleaning for 2021 data
# replace all dashes with 0
col_2021 = ['HCL_S4', 'HCL_S3', 'HCL_S2', 'HCL_S1',
            'CL_S4', 'CL_S3', 'CL_S2', 'CL_S1',
            'ML_S4', 'ML_S3', 'ML_S2', 'ML_S1',
            'TL_S3', 'TL_S2', 'TL_S1',
            'CL_O', 'ML_O', 'TL_O']
df_S4_2021[col_2021] = df_S4_2021[col_2021].replace({'-': 0})
# turn columns into numeric values
for i in col_2021:
    df_S4_2021[i] = pd.to_numeric(df_S4_2021[i])


# In[17]:


# Merging all dfs together (Row order is preserved)
dfs = [df_S4_2016, df_S4_2017, df_S4_2018, df_S4_2019, df_S4_2020, df_S4_2021]
combined_df = pd.concat(dfs)
combined_df = combined_df.reset_index(drop=True)

# Standardize nan
combined_df[cols] = combined_df[cols].replace({'0': np.nan, 0: np.nan})

# Setting boolean for carded athlete
col = 'Sec4_CardingLevel'
combined_df[col] = combined_df[col].replace(np.nan, False)
combined_df[col] = combined_df[col].replace(
    ['L3', 'Y', 'L4P', 'L4', 'YT', 'TL3', 'E3', 'B4', 'ET3', 'Y+'], True)


# In[18]:


# Seperating df based on languages
combined_df_HCL = combined_df[combined_df['HCL_S1'].notna()]
combined_df_CL = combined_df[combined_df['CL_S1'].notna()]
combined_df_ML = combined_df[combined_df['ML_S1'].notna()]
combined_df_TL = combined_df[combined_df['TL_S1'].notna()]

# Drop empty columns
combined_df_HCL = combined_df_HCL[['Serial number', 'Gender', 'Sec4_CardingLevel', 'Course',
                                  'TScore', 'MT', 'HMT', 'ResidentialType', 'HCL_S4', 'HCL_S3', 'HCL_S2', 'HCL_S1']]
combined_df_CL = combined_df_CL[['Serial number', 'Gender', 'Sec4_CardingLevel', 'Course',
                                 'TScore', 'MT', 'ResidentialType', 'CL_S4', 'CL_S3', 'CL_S2', 'CL_S1', 'CL_O']]
combined_df_ML = combined_df_ML[['Serial number', 'Gender', 'Sec4_CardingLevel', 'Course',
                                 'TScore', 'MT', 'ResidentialType', 'ML_S4', 'ML_S3', 'ML_S2', 'ML_S1', 'ML_O']]
combined_df_TL = combined_df_TL[['Serial number', 'Gender', 'Sec4_CardingLevel', 'Course',
                                 'TScore', 'MT', 'ResidentialType', 'TL_S4', 'TL_S3', 'TL_S2', 'TL_S1', 'TL_O']]

print(combined_df_HCL.shape, combined_df_CL.shape,
      combined_df_ML.shape, combined_df_TL.shape)

# Current Non-NaN value
print(combined_df_CL['CL_S4'].notnull().sum())
print(combined_df_ML['ML_S4'].notnull().sum())
print(combined_df_TL['TL_S4'].notnull().sum())


# In[19]:


# Overwrite S4 MT language with O level result
combined_df_CL.loc[combined_df_CL['CL_O'].notna(), ['CL_S4']
                   ] = combined_df_CL['CL_O']
combined_df_ML.loc[combined_df_ML['ML_O'].notna(), ['ML_S4']
                   ] = combined_df_ML['ML_O']
combined_df_TL.loc[combined_df_TL['TL_O'].notna(), ['TL_S4']
                   ] = combined_df_TL['TL_O']

# We will proceed without (HML, Bahasa Indonesia, Basic Chinese) given low data, and MT O level score (Only 2019 DF has)

print(combined_df_HCL.shape, combined_df_CL.shape,
      combined_df_ML.shape, combined_df_TL.shape)

# Validation that the Non-NaN value has increased
print(combined_df_CL['CL_S4'].notnull().sum())
print(combined_df_ML['ML_S4'].notnull().sum())
print(combined_df_TL['TL_S4'].notnull().sum())


# In[20]:


# Drop O level since it's no longer needed
combined_df_CL.drop(columns=['CL_O'], inplace=True)
combined_df_ML.drop(columns=['ML_O'], inplace=True)
combined_df_TL.drop(columns=['TL_O'], inplace=True)

# Reset index
combined_df_HCL = combined_df_HCL.reset_index(drop=True)
combined_df_CL = combined_df_CL.reset_index(drop=True)
combined_df_ML = combined_df_ML.reset_index(drop=True)
combined_df_TL = combined_df_TL.reset_index(drop=True)

# Standardize the NaN
combined_df_HCL = combined_df_HCL.replace(
    {'0': np.nan, 0: np.nan, '-': np.nan, '': np.nan})
combined_df_CL = combined_df_CL.replace({'0': np.nan, 0: np.nan, '-': np.nan})
combined_df_ML = combined_df_ML.replace({'0': np.nan, 0: np.nan, '-': np.nan})
combined_df_TL = combined_df_TL.replace({'0': np.nan, 0: np.nan, '-': np.nan})


# In[21]:


combined_df_HCL.head()


# In[22]:


combined_df.to_csv("Combined.csv")


# ## DataFrame:
# 
#     Merging P6 Screening Result with Secondary School Result
#     P6 2016 + Secondary 2020
#     P6 2017 + Secondary 2021
# 

# In[23]:


# Drop un-used columns
data_P6_2016.drop(list(data_P6_2016)[102::], axis=1, inplace=True)
data_P6_2017.drop(list(data_P6_2017)[103::], axis=1, inplace=True)

# Drop un-named columns
data_P6_2016 = data_P6_2016.loc[:, ~
                                data_P6_2016.columns.str.contains('^Unnamed')]
data_P6_2017 = data_P6_2017.loc[:, ~
                                data_P6_2017.columns.str.contains('^Unnamed')]

# Drop 'x' ID rows
data_P6_2016 = data_P6_2016.loc[data_P6_2016["Serial number"] != 'x']
data_P6_2017 = data_P6_2017.loc[data_P6_2017["Serial number"] != 'x']

# Drop 'Nan' ID rows and Data that is empty
data_P6_2016 = data_P6_2016.dropna()
data_P6_2017 = data_P6_2017.dropna()

# Keep only essential columns
data_P6_2016 = data_P6_2016[['Serial number', 'Q1 M', 'Q1F', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7',
                            'Q8a', 'Q8b', 'Q8c', 'Percentage (%)', 'Percentage (%).1', 'Percentage (%).2', 'Percentage (%).3', 'Percentage (%).4', 'Percentage (%).5',
                             'Q1.6', 'Q2a', 'Q2b', 'Q2c', 'Q2d', 'Q2e', 'Q2f', 'Q2g', 'Q2h', 'Q2i', 'Q2j', 'Q2k', 'Q3.7', 'Q4a', 'Q4b', 'Q4c', 'Q4d', 'Q4e', 'Q4f',
                             'Q4g', 'Q4h', 'Q4i', 'Q4j', 'Q4k']]
data_P6_2017 = data_P6_2017[['Serial number', 'Q1 M', 'Q1F', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7',
                            'Q8a', 'Q8b', 'Q8c', 'Percentage (%)', 'Percentage (%).1', 'Percentage (%).2', 'Percentage (%).3', 'Percentage (%).4', 'Percentage (%).5',
                             'Q1.6', 'Q2a', 'Q2b', 'Q2c', 'Q2d', 'Q2e', 'Q2f', 'Q2g', 'Q2h', 'Q2i', 'Q2j', 'Q2k', 'Q3.7', 'Q4a', 'Q4b', 'Q4c', 'Q4d', 'Q4e', 'Q4f',
                             'Q4g', 'Q4h', 'Q4i', 'Q4j', 'Q4k']]


# In[73]:


# Convert the column to type float so that we can merge
data_P6_2016['Serial number'] = data_P6_2016['Serial number'].astype(int)
data_P6_2017['Serial number'] = data_P6_2017['Serial number'].astype(int)

# Combined both data
# Merge the dataframe
df_psych_original = pd.concat([data_P6_2016, data_P6_2017])

# Merging of data (Paring of S/No. is unique)
df_P6_HCL = df_psych_original.merge(
    combined_df_HCL, how='inner', on=['Serial number'])
df_P6_CL = df_psych_original.merge(
    combined_df_CL, how='inner', on=['Serial number'])
df_P6_ML = df_psych_original.merge(
    combined_df_ML, how='inner', on=['Serial number'])
df_P6_TL = df_psych_original.merge(
    combined_df_TL, how='inner', on=['Serial number'])

print(df_P6_HCL.shape, df_P6_CL.shape, df_P6_ML.shape, df_P6_TL.shape)
# Upon merging, the row size will shrink as only those with the matching serial no. will retain
df_P6_CL.head()


# # Data Visualization
# 
# ---
# 

# In[25]:


# Set up matplotlib figure with Two subplots
f, axes = plt.subplots(4, figsize=(24, 12))
xticks = [0, 50, 100, 150, 200, 250, 300]

axes[0].set_title('combined_df_HCL')
g1 = sb.boxplot(data=combined_df_HCL.iloc[:, 4:], orient="h", ax=axes[0])
axes[1].set_title('combined_df_CL')
g2 = sb.boxplot(data=combined_df_CL.iloc[:, 4:], orient="h", ax=axes[1])
axes[2].set_title('combined_df_ML')
g1 = sb.boxplot(data=combined_df_ML.iloc[:, 4:], orient="h", ax=axes[2])
axes[3].set_title('combined_df_TL')
g2 = sb.boxplot(data=combined_df_TL.iloc[:, 4:], orient="h", ax=axes[3])


# In[26]:


f, axes = plt.subplots(4, figsize=(24, 18))

axes[0].set_title('combined_df_HCL')
sb.heatmap(data=combined_df_HCL.corr(), vmin=-1, vmax=1,
           annot=True, fmt=".2f",  cmap="RdBu", ax=axes[0])
axes[1].set_title('combined_df_CL')
sb.heatmap(data=combined_df_CL.corr(), vmin=-1, vmax=1,
           annot=True, fmt=".2f",  cmap="RdBu", ax=axes[1])
axes[2].set_title('combined_df_ML')
sb.heatmap(data=combined_df_ML.corr(), vmin=-1, vmax=1,
           annot=True, fmt=".2f",  cmap="RdBu", ax=axes[2])
axes[3].set_title('combined_df_TL')
sb.heatmap(data=combined_df_TL.corr(), vmin=-1, vmax=1,
           annot=True, fmt=".2f",  cmap="RdBu", ax=axes[3])


# # Functions
# 
# ---
# 

# In[27]:


# Getting results band
def getresults(df, subject):
    data = df[subject]
    lst = []
    for score in data:
        if score >= 75:
            lst.append(1)
        elif score >= 70 and score < 75:
            lst.append(2)
        elif score >= 65 and score < 70:
            lst.append(3)
        elif score >= 60 and score < 65:
            lst.append(4)
        elif score >= 55 and score < 60:
            lst.append(5)
        elif score >= 50 and score < 55:
            lst.append(6)
        elif score >= 45 and score < 50:
            lst.append(7)
        elif score >= 40 and score < 45:
            lst.append(8)
        elif score < 40 and score > 0:
            lst.append(9)
        else:  # catches NaNs and value 0
            lst.append(float("NaN"))
    return lst


# In[28]:


def getresults_arr(score):
    if score >= 75:
        return(1)
    elif score >= 70 and score < 75:
        return(2)
    elif score >= 65 and score < 70:
        return(3)
    elif score >= 60 and score < 65:
        return(4)
    elif score >= 55 and score < 60:
        return(5)
    elif score >= 50 and score < 55:
        return(6)
    elif score >= 45 and score < 50:
        return(7)
    elif score >= 40 and score < 45:
        return(8)
    elif score < 40:
        return(9)
    else:  # catches NaNs and value 0
        return(float("NaN"))


# In[29]:


# PSLE HMT tier
def getHMT_psle(df):
    data = df['HMT']
    lst = []
    for score in data:
        if score == 'DIST':
            lst.append(1)
        elif score == 'MERIT':
            lst.append(2)
        elif score == 'PASS':
            lst.append(3)
        elif score == 'UNGRADED':
            lst.append(4)
        else:
            lst.append(5)
    return lst


# In[30]:


# get psle results band
def encode_psle(df, subject):
    data = df[subject]
    lst = []
    for score in data:
        if score == 'A*':
            lst.append(1)
        elif score == 'A':
            lst.append(2)
        elif score == 'B':
            lst.append(3)
        elif score == 'C':
            lst.append(4)
        elif score == 'D':
            lst.append(5)
        elif score == 'E':
            lst.append(6)
        elif score == 'F':
            lst.append(7)
        else:
            lst.append(float("NaN"))
    return lst


# In[31]:


# get housing tier
def gethousing(df):
    data = df['ResidentialType']
    lst = []
    for house in data:
        if house == 'Detached House':
            lst.append(1)
        elif house == 'Semi-Detached House':
            lst.append(2)
        elif house == 'Terrace':
            lst.append(3)
        elif house == 'Private Flat/Apartment':
            lst.append(4)
        elif house == 'Govt/Quasi-Govt Executive Flat':
            lst.append(5)
        elif house == 'HDB/SAF/PSA/PUB 5_Room Flat':
            lst.append(6)
        elif house == 'HDB/SAF/PSA/PUB 4_Room Flat':
            lst.append(7)
        elif house == 'HDB/SAF/PSA/PUB 3_Room Flat':
            lst.append(8)
        elif house == 'other':
            lst.append(9)
        else:
            lst.append(float("NaN"))
    return lst


# In[32]:


def getcourse_tier(df):
    data = df['Course']
    lst = []
    for course in data:
        if course == 'Express':
            lst.append(1)
        elif course == 'Normal Academic':
            lst.append(2)
        elif course == 'Normal Technical':
            lst.append(3)
        else:
            lst.append(float("NaN"))
    return lst


# In[33]:


def getGender(df):
    data = df['Gender']
    lst = []
    for gender in data:
        if gender == 'Male':
            lst.append(1)
        elif gender == 'Female':
            lst.append(0)
        else:
            lst.append(float("NaN"))
    return lst


# In[34]:


def getCarding(df):
    data = df['Sec4_CardingLevel']
    lst = []
    for i in data:
        if i == True:
            lst.append(1)
        elif i == False:
            lst.append(0)
        else:
            lst.append(float('NaN'))
    return lst


# In[35]:


def leeway(pred_res, allowance):
    lst = []
    for i in pred_res:
        actual = getresults_arr(i)
        plus = getresults_arr(i + allowance)
        minus = getresults_arr(i - allowance)
        lst.append([actual, plus, minus])
    return lst


# In[36]:


from sklearn.impute import KNNImputer
# Comment: Simple Imputer will yield the same result


def KNN_Imputer(df, cols):
    imputer = KNNImputer(n_neighbors=3)
    for col in cols:
        df[col] = imputer.fit_transform(df[[col]])
        df[col] = np.around(df[col], 0)
    return df


# # Remove Outlier

# In[ ]:


cols = ['HCL_S4', 'HCL_S3', 'HCL_S2', 'HCL_S1']

# Calculate the quartiles
Q1 = combined_df[cols].quantile(0.25)
Q3 = combined_df[cols].quantile(0.75)

# Rule to identify outliers
IQR = Q3 - Q1

combined_df_HCL = combined_df_HCL[~((combined_df_HCL[cols] < (
    Q1 - 1.5 * IQR)) | (combined_df_HCL[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
df_P6_HCL = df_P6_HCL[~((df_P6_HCL[cols] < (
    Q1 - 1.5 * IQR)) | (df_P6_HCL[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

########################################################################################
cols = ['CL_S4', 'CL_S3', 'CL_S2', 'CL_S1']
# Calculate the quartiles
Q1 = combined_df[cols].quantile(0.25)
Q3 = combined_df[cols].quantile(0.75)

# Rule to identify outliers
IQR = Q3 - Q1

combined_df_CL = combined_df_CL[~((combined_df_CL[cols] < (
    Q1 - 1.5 * IQR)) | (combined_df_CL[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
df_P6_CL = df_P6_CL[~((df_P6_CL[cols] < (
    Q1 - 1.5 * IQR)) | (df_P6_CL[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

########################################################################################
cols = ['ML_S4', 'ML_S3', 'ML_S2', 'ML_S1']
# Calculate the quartiles
Q1 = combined_df[cols].quantile(0.25)
Q3 = combined_df[cols].quantile(0.75)

# Rule to identify outliers
IQR = Q3 - Q1

combined_df_ML = combined_df_ML[~((combined_df_ML[cols] < (
    Q1 - 1.5 * IQR)) | (combined_df_ML[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
df_P6_ML = df_P6_ML[~((df_P6_ML[cols] < (
    Q1 - 1.5 * IQR)) | (df_P6_ML[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# # Linear Regression
# 

# In[37]:


df_linreg = combined_df_ML.copy()

# To convert all these data into numerical value for decision tree
# Categorical data to scoring system
df_linreg['Gender'] = getGender(combined_df_ML)
df_linreg['Sec4_CardingLevel'] = getCarding(combined_df_ML)
df_linreg['Course'] = getcourse_tier(combined_df_ML)
df_linreg['ResidentialType'] = gethousing(combined_df_ML)
# Numerical data to scoring system
df_linreg['MT'] = encode_psle(combined_df_ML, 'MT')
# df_linreg['HMT'] = getHMT_psle(combined_df_ML)
# df_linreg = df_linreg[df_linreg['ML_S4'].notna()]
# df_linreg = df_linreg[df_linreg['ML_S3'].notna()]
# df_linreg = df_linreg[df_linreg['ML_S2'].notna()]
# df_linreg = df_linreg[df_linreg['ML_S1'].notna()]
# Columns that are applying KNN Imputer
cols = df_linreg.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
df_linreg = KNN_Imputer(df_linreg, cols)
df_linreg = pd.DataFrame(df_linreg, columns=cols)

# Validation
print(df_linreg.shape)


# In[38]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler()
# scaler.fit_transform()  # scales data
# scaler.inverse_transform()  # transforms scaled data back into original form
X = df_linreg[['TScore', 'Gender', 'Sec4_CardingLevel',
               'Course', 'MT', 'ResidentialType', 'ML_S3', 'ML_S2', 'ML_S1']]
y = df_linreg[['ML_S4']]
# X = X.to_numpy()
# y = y.to_numpy()

X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=2)
y_test_final = scaler.inverse_transform(y_test).flatten()


# In[39]:


# Linear regression model
from sklearn.linear_model import LinearRegression
regmodel = LinearRegression().fit(X_train, y_train)


# In[40]:


# converting test results into grades
y_test_grade = []
for i in y_test_final:
    y_test_grade.append(getresults_arr(i))


# In[41]:


# Predictions from model
lr_preds = regmodel.predict(X_test)
# lr_preds = lr_preds.reshape(-1, 1)
lr_preds = scaler.inverse_transform(lr_preds).flatten().round()


# In[42]:


lr_preds_rg = leeway(lr_preds, 3)


# In[43]:


correct = 0
for i in range(len(lr_preds)):
    if y_test_grade[i] in lr_preds_rg[i]:
        correct += 1
print(correct / len(lr_preds))


# # Linear Regression for P6 Data
# 

# In[44]:


df_test1 = df_P6_HCL.copy()
df_test1['Gender'] = getGender(df_P6_HCL)
df_test1['Sec4_CardingLevel'] = getCarding(df_P6_HCL)
df_test1['Course'] = getcourse_tier(df_P6_HCL)
df_test1['ResidentialType'] = gethousing(df_P6_HCL)

# Numerical data to scoring system
df_test1['MT'] = encode_psle(df_P6_HCL, 'MT')
df_test1['HMT'] = getHMT_psle(df_P6_HCL)
# df_test1 = df_test1[df_test1['HCL_S4'].notna()]
# df_test1 = df_test1[df_test1['HCL_S3'].notna()]
# df_test1 = df_test1[df_test1['HCL_S2'].notna()]
# df_test1 = df_test1[df_test1['HCL_S1'].notna()]

# Columns that are applying KNN Imputer
cols = df_test1.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
df_test1 = KNN_Imputer(df_test1, cols)
df_test1 = pd.DataFrame(df_test1, columns=cols)

# Reset the df index
df_test1.reset_index(inplace=True)
print(df_test1.shape)
df_test1.head(5)


# In[45]:


X = df_test1.copy()
print(X.shape)
X.drop(columns=['HCL_S4'], inplace=True)
Y = y = df_test1[['HCL_S4']]
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=2)
y_test_final = scaler.inverse_transform(y_test).flatten()


# In[46]:


regmodel_p6 = LinearRegression().fit(X_train, y_train)


# In[47]:


# converting test results into grades
y_test_grade = []
for i in y_test_final:
    y_test_grade.append(getresults_arr(i))


# In[48]:


# Predictions from model
lr_predsp6 = regmodel_p6.predict(X_test)
# lr_predsp6 = lr_predsp6.reshape(-1, 1)
lr_predsp6 = scaler.inverse_transform(lr_predsp6).flatten().round()


# In[49]:


lr_predsp6_rg = leeway(lr_predsp6, 3)


# In[50]:


correct = 0
for i in range(len(lr_predsp6)):
    if y_test_grade[i] in lr_predsp6_rg[i]:
        correct += 1
print(correct / len(lr_predsp6))


# # Decision Tree
# 
# ---
# 
#     Supervised Machine Learmomg
# 
# Decision tree builds regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed
# 

# Data used: overall Dataset.
# Dataset is too small to be split based on streaming. (i.e. 236 express, 49 na)
# 

# In[51]:


# Temp to do the switch between languages
# var = combined_df_HCL
# var = combined_df_CL
var = combined_df_ML

# Make a duplicate dataframe for manipulation
df_decisionTree_combined = var.copy()

# Drop serial number column as it is useless
df_decisionTree_combined.drop(columns=['Serial number'], inplace=True)

# Columns that are applying KNN Imputer
cols = df_decisionTree_combined.columns.to_numpy()

# To convert all these data into numerical value for decision tree
# Categorical data to scoring system
df_decisionTree_combined['Gender'] = getGender(var)
df_decisionTree_combined['Sec4_CardingLevel'] = getCarding(var)
df_decisionTree_combined['Course'] = getcourse_tier(var)
df_decisionTree_combined['ResidentialType'] = gethousing(var)

# Numerical data to scoring system
df_decisionTree_combined['MT'] = encode_psle(var, 'MT')
if 'HMT' in cols:
    df_decisionTree_combined['HMT'] = getHMT_psle(var)

# Banding System
df_decisionTree_combined['ML_S4'] = getresults(
    df_decisionTree_combined, 'ML_S4')
df_decisionTree_combined['ML_S3'] = getresults(
    df_decisionTree_combined, 'ML_S3')
df_decisionTree_combined['ML_S2'] = getresults(
    df_decisionTree_combined, 'ML_S2')
df_decisionTree_combined['ML_S1'] = getresults(
    df_decisionTree_combined, 'ML_S1')

# Fill up the NaN based on KNN Imputer
df_decisionTree_combined = KNN_Imputer(df_decisionTree_combined, cols)
df_decisionTree_combined = pd.DataFrame(df_decisionTree_combined, columns=cols)

# Validation
print(df_decisionTree_combined.shape)
df_decisionTree_combined.head()


# In[52]:


# Temp to do the switch between languages
# var = df_P6_HCL
# var = df_P6_CL
var = df_P6_ML

# Make a duplicate dataframe for manipulation
df_psych = var.copy()
df_psych.drop(columns=['Serial number'], inplace=True)

# To convert all these data into numerical value for decision tree
# Categorical data to scoring system
df_psych['Gender'] = getGender(var)
df_psych['Sec4_CardingLevel'] = getCarding(var)
df_psych['Course'] = getcourse_tier(var)
df_psych['ResidentialType'] = gethousing(var)

# Columns that are applying KNN Imputer
cols = df_psych.columns.to_numpy()

# Numerical data to scoring system
df_psych['MT'] = encode_psle(var, 'MT')
if 'HMT' in cols:
    df_psych['HMT'] = getHMT_psle(var)

# Ranking system
df_psych['ML_S4'] = getresults(df_psych, 'ML_S4')
df_psych['ML_S3'] = getresults(df_psych, 'ML_S3')
df_psych['ML_S2'] = getresults(df_psych, 'ML_S2')
df_psych['ML_S1'] = getresults(df_psych, 'ML_S1')

# Fill up the NaN based on KNN Imputer
df_psych = KNN_Imputer(df_psych, cols)
df_psych = pd.DataFrame(df_psych, columns=cols)

# # Validation
print(df_psych.shape)
df_psych.head()


# In[53]:


# Import Decision Tree Classifier model from Scikit-Learn
from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree Classifier object (Do experiment with the max_depth)
dectree_combined = DecisionTreeClassifier(max_depth=2)
dectree_psych = DecisionTreeClassifier(max_depth=2)


# **Observation** : Note that for the data, Categorical variables are qualitative and it can be "encoded" to Nominal, Ordinal, or Interval.
# If the data has relationship between one another (E.g., Residential Estate => Economical Status), we gave it a rank. Whereas for (E.g., Academy => There's no relationship between one to another) we used One-Hot-Encoder
# 
# Note that this table is created _manually_, and this is my interpretation. Feel free to choose your own.
# 
# | Variable          | Observation           | Description                                                                                            |
# | :---------------- | :-------------------- | :----------------------------------------------------------------------------------------------------- |
# | Gender            | Categorial : Nominal  | Changed to numeric (0,1) variable with no implication of quantitiative value or order                  |
# | Sec4_CardingLevel | Categorial : Nominal  | Changed to boolean (False,True) variable with no implication of quantitiative value or order           |
# | Race              | Categorial : Nominal  | Used One-Hot-Encoder to label variable with no implication of quantitiative value or order             |
# | Academy           | Categorial : Nominal  | Used One-Hot-Encoder to label variable with no implication of quantitiative value or order             |
# | Course            | Categorial : Nominal  | Changed to numeic (1,2,3) to label variable with no implication of quantitiative value or order        |
# | Maths             | Categorial : Interval | Re-indexed Letter-Grade to Band to represent a numeric scale where the order and differences are known |
# | ResidentialType   | Categorial : Ordinal  | Labeled variable with a scale 1-to-8 for social-economical impact                                      |
# | TScore            | Numerical : Ratio     | Used original range of 0 to 300                                                                        |
# | AM_S4             | Numerical : Ratio     | Re-indexed to the last char of A1,A2,B3... banding system                                              |
# | AM_S3             | Numerical : Ratio     | Re-indexed to the last char of A1,A2,B3... banding system                                              |
# | EM_S4             | Numerical : Ratio     | Re-indexed to the last char of A1,A2,B3... banding system                                              |
# | EM_S3             | Numerical : Ratio     | Re-indexed to the last char of A1,A2,B3... banding system                                              |
# | EM_S2             | Numerical : Ratio     | Re-indexed to the last char of A1,A2,B3... banding system                                              |
# | EM_S1             | Numerical : Ratio     | Re-indexed to the last char of A1,A2,B3... banding system                                              |
# 

# In[54]:


from sklearn.model_selection import train_test_split
# Extract Response and Predictors
X_combined = df_decisionTree_combined.drop(['ML_S4'], axis=1,
                                           inplace=False)  # Copy of df without predictor
y_combined = pd.DataFrame(df_decisionTree_combined['ML_S4'])


# Split the Dataset into random Train and Test
X_combined_train, X_combined_test, y_combined_train, y_combined_test = train_test_split(
    X_combined, y_combined, test_size=0.3, random_state=2)

# Fit the model
dectree_combined.fit(X_combined_train, y_combined_train)

###############################################################################################
# Extract Response and Predictors
X_psych = df_psych.drop(['ML_S4'], axis=1, inplace=False)
y_psych = pd.DataFrame(df_psych['ML_S4'])


# Split the Dataset into random Train and Test
X_psych_train, X_psych_test, y_psych_train, y_psych_test = train_test_split(
    X_psych, y_psych, test_size=0.2, random_state=2)

# Fit the model
dectree_psych.fit(X_psych_train, y_psych_train)


# In[55]:


# Plot the trained Decision Tree
from sklearn.tree import plot_tree

f = plt.figure(figsize=(24, 12))
# Left arrow: True
# Right arrow: False
# @para: class_names=['1', '2', '3', '4', '5', '6', '7', '8', '9']
plot_tree(dectree_combined, fontsize=10, filled=True, rounded=True,
          feature_names=X_combined_train.columns)


# In[56]:


# Plot the trained Decision Tree
from sklearn.tree import plot_tree

f = plt.figure(figsize=(24, 12))
# Left arrow: True
# Right arrow: False
# @para: class_names=['1', '2', '3', '4', '5', '6', '7', '8', '9']
plot_tree(dectree_combined, fontsize=10, filled=True, rounded=True,
          feature_names=X_combined_train.columns)


# #### Prediction on Test Data and Goodness of Fit
# 
# Check how good the predictions are on the Test Set.  
# Metrics : Classification Accuracy and Confusion Matrix.
# 

# In[57]:


# Predict the Response corresponding to Predictors
y_combined_test_pred = dectree_combined.predict(X_combined_test)
y_psych_test_pred = dectree_psych.predict(X_psych_test)


# In[58]:


list_test = y_combined_test.to_numpy()
list_pred = y_combined_test_pred
correct = 0
for i in range(len(list_test)):
    if abs(list_test[i] - list_pred[i]) <= 1:  # Tweak val to change the accuracy
        correct += 1
print("Test Combined")
print("Accuracy : ", correct/len(list_test))


# In[59]:


list_test = y_psych_test.to_numpy()
list_pred = y_psych_test_pred
correct = 0
for i in range(len(list_test)):
    if abs(list_test[i] - list_pred[i]) <= 1:  # Tweak val to change the accuracy
        correct += 1
print("Test Psych")
print("Accuracy : ", correct/len(list_test))


# #### Comment
# 
# **Important** : Note the huge imbalance in the _False Positives_ and _False Negatives_ in the confusion matrix. _False Negatives_ are much higher in number than _False Positives_ in both Train and Test data. This is not surprising -- actually, this is a direct effect of the huge number of classes.
# 

# In[ ]:




# # DNN Regressor
# 
# ---
# 
#     Deep Neural Network to predict numeric value (English_S4)
# 

# Classify structured data with feature columns
# 

# In[60]:


# Library to rescale variables into the range [0,1]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Supress warning for SettingWithCopyError
pd.options.mode.chained_assignment = None

# Temp to do the switch between languages
# var = combined_df_HCL
# var = combined_df_CL
var = combined_df_ML

# Dataset
df_DNN = var.copy().dropna()

# Columns that are applying KNN Imputer
cols = df_DNN.columns.to_numpy()

# Numerical data to scoring system
df_DNN['MT'] = encode_psle(df_DNN, 'MT')
if 'HMT' in cols:
    df_DNN['HMT'] = getHMT_psle(df_DNN)
    df_DNN['HMT'] = pd.to_numeric(df_DNN['HMT'])
# Convert from boolean to float
df_DNN['Sec4_CardingLevel'] = df_DNN['Sec4_CardingLevel'].astype(int)

# Banding System
df_DNN['ML_S3'] = getresults(df_DNN, 'ML_S3')
df_DNN['ML_S2'] = getresults(df_DNN, 'ML_S2')
df_DNN['ML_S1'] = getresults(df_DNN, 'ML_S1')

# Scale continuous variable in the dataset
df_DNN['TScore'] = scaler.fit_transform(df_DNN['TScore'].values.reshape(-1, 1))
df_DNN['ML_S3'] = scaler.fit_transform(df_DNN['ML_S3'].values.reshape(-1, 1))
df_DNN['ML_S2'] = scaler.fit_transform(df_DNN['ML_S2'].values.reshape(-1, 1))
df_DNN['ML_S1'] = scaler.fit_transform(df_DNN['ML_S1'].values.reshape(-1, 1))
# df_DNN['HCL_S4'] = scaler.fit_transform(df_DNN['HCL_S4'].values.reshape(-1, 1))

# Extract Response and Predictors
X = df_DNN.drop(columns=['ML_S4'],
                inplace=False)  # Copy of df without predictor
y = pd.DataFrame(df_DNN['ML_S4'])

# X needs to be in dataframe, y needs to be in series
y = y.squeeze()

# Split the Dataset into random Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[61]:


# # Library to rescale variables into the range [0,1]
# from sklearn.preprocessing import MinMaxScaler

# # Supress warning for SettingWithCopyError
# pd.options.mode.chained_assignment = None

# scaler = MinMaxScaler()

# # Dataset
# df_DNN = df_psych_original.copy().dropna()
# # Extract Response and Predictors
# X = df_DNN.drop(columns=['English_S4'], inplace=False)
# # Convert Boolean to Integer (0,1)
# X['Sec4_CardingLevel'] = X['Sec4_CardingLevel'].astype(int)
# y = df_DNN.iloc[:, df_DNN.columns.get_loc("English_S4")]

# # Scale continuous variable in the dataset
# list_col = X.columns.tolist()
# list_rmv = ['Gender', 'Sec4_CardingLevel', 'Course', 'EL', 'ResidentialType', 'Race', 'Academy']
# for ele in list_rmv:
#     list_col.remove(ele)
# for ele in list_col:
#     X[ele] = scaler.fit_transform(X[ele].values.reshape(-1, 1))
# # Remove % symbol
# X = X.rename(columns={'Percentage (%)': 'Percentage',
#                       'Percentage (%).1': 'Percentage1',
#                       'Percentage (%).2': 'Percentage2',
#                       'Percentage (%).3': 'Percentage3',
#                       'Percentage (%).4': 'Percentage4',
#                       'Percentage (%).5': 'Percentage5',
#                       'Q1 M': 'Q1_M',
#                       'Q1.6': 'Q1_6',
#                       'Q3.7': 'Q3_7',
#                       'Percentage (%).5': 'Percentage5',
#                       })
# # for col in X.columns:
# #     print(col)
# X.drop(columns=['index'], inplace=True)
# print(len(X.columns))
# # y = scaler.fit_transform(y.values.reshape(-1, 1))

# # Split the Dataset into random Train and Test
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=2)


# In[62]:


Gender_vocab = ['Male', 'Female']
Gender_column = tf.feature_column.categorical_column_with_vocabulary_list(
    key="Gender", vocabulary_list=Gender_vocab)

Course_vocab = ['Express', 'Normal Academic', 'Normal Technical']
Course_column = tf.feature_column.categorical_column_with_vocabulary_list(
    key="Course", vocabulary_list=Course_vocab)

ResidentialType_vocab = ['HDB/SAF/PSA/PUB 4_Room Flat', 'HDB/SAF/PSA/PUB 5_Room Flat', 'Private Flat/Apartment',
                         'Govt/Quasi-Govt Executive Flat', 'HDB/SAF/PSA/PUB 3_Room Flat', 'Terrace',
                         'Semi-Detached House', 'Other', 'Detached House']
ResidentialType_column = tf.feature_column.categorical_column_with_vocabulary_list(
    key="ResidentialType", vocabulary_list=ResidentialType_vocab)


# Adding all features to a list
# 

# In[63]:


# Columns
feature_columns = [
    tf.feature_column.indicator_column(Gender_column),
    tf.feature_column.indicator_column(Course_column),
    tf.feature_column.indicator_column(ResidentialType_column),
    tf.feature_column.numeric_column(
        key='Sec4_CardingLevel', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='HMT', dtype=tf.float64),
    tf.feature_column.numeric_column(key='MT', dtype=tf.float64),
    tf.feature_column.numeric_column(key='TScore', dtype=tf.float64),
    tf.feature_column.numeric_column(key='ML_S1', dtype=tf.float64),
    tf.feature_column.numeric_column(key='ML_S2', dtype=tf.float64),
    tf.feature_column.numeric_column(key='ML_S3', dtype=tf.float64),

    ###################
    # tf.feature_column.numeric_column(key='Percentage', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Percentage1', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Percentage2', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Percentage3', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Percentage4', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Percentage5', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q1_M', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q1F', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q2', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q3', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q4', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q5', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q6', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q7', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q8a', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q8b', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q8c', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q1_6', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q2a', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q2b', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q2c', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q2d', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q2e', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q2f', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q2g', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q2h', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q2i', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q2j', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q2k', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q3_7', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q4a', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q4b', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q4c', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q4d', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q4e', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q4f', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q4g', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q4h', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q4i', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q4j', dtype=tf.float64),
    # tf.feature_column.numeric_column(key='Q4k', dtype=tf.float64),
]


# Creates the input function for the estimator object
# 

# In[64]:


# Reset the dataframe index
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=100, num_epochs=10,
                                                           shuffle=False)


# Create the estimator model
# 
#     3 layers, each has 32 neurons;
#     Adam as an optimizer (learning rate = 0.001)
#     Relu as an activation function
# 

# In[65]:


model = tf.estimator.DNNRegressor(hidden_units=[32, 32, 32], feature_columns=feature_columns,
                                  optimizer=tf.optimizers.Adam(
                                      learning_rate=0.1),
                                  activation_fn=tf.nn.relu)


# In[66]:


model.train(input_fn=input_func, steps=10000)


# Create a prediction input function and then use the .predict method to create a list or predictions on a test data.
# 

# In[67]:


predict_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=X_test,
    batch_size=100,
    num_epochs=1,
    shuffle=False)

pred_gen = model.predict(predict_input_func)
predictions = list(pred_gen)

final_preds = []
for pred in predictions:
    final_preds.append(pred['predictions'])


# ### Evaluating a model
# 
# ---
# 
#     Loss/Cost Function
# 
# As we now know our neural network feeds information through the layers until it eventually reaches an output layer. This layer contains the results that we look at to determine the prediciton from our network. In the training phase it is likely that our network will make many mistakes and poor predicitions. In fact, at the start of training our network doesn't know anything (it has random weights and biases)!
# 
# We need some way of evaluating if the network is doing well and how well it is doing. For our training data we have the features (input) and the labels (expected output), because of this we can compare the output from our network to the expected output. Based on the difference between these values we can determine if our network has done a good job or poor job. If the network has done a good job, we'll make minor changes to the weights and biases. If it has done a poor job our changes may be more drastic.
# 
# So, this is where the cost/loss function comes in. This function is responsible for determining how well the network did. We pass it the output and the expected output, and it returns to us some value representing the cost/loss of the network. This effectively makes the networks job to optimize this cost function, trying to make it as low as possible.
# 
#     Check how good the predictions are on the Test Set.
#     Metric : Explained Variance or R^2 on the Test Set.
#     Metric : Mean Squared Error (MSE) on the Test Set
# 

# In[68]:


# Goodness of Fit of the DDN Model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print("Mean Squared Error: ", mean_squared_error(y_test, final_preds))
print("Mean Absolute Error: ", mean_absolute_error(y_test, final_preds))
print("R^2 Error: ", r2_score(y_test, final_preds))


# Comment: Mean absolute value shows that on average, the model makes a mistake of around 5~12 marks. This is considered as a poor result. To improve the model, consider inputing more data. Banding can also be an alternative when we're viewing the data.
# 

# Compare real values to predicted
# 

# In[69]:


list_pred = []
for num in final_preds:
    list_pred.append(num[0])

d = {'Actual Result': y_test, 'Predicted by Model': list_pred}
df = pd.DataFrame(data=d)
df.round(2)[:10]  # Show 10 results


# In[70]:


correct = 0
list_y = y_test.to_numpy()
for i in range(len(list_pred)):
    # allowance of 1 increases prediction accuracy
    if abs(list_pred[i] - list_y[i]) <= 5:
        correct += 1
print(correct / len(list_pred))


# In[ ]:




# # Multi Layer Perceptron
# 
# ---
# 
#     Feedforward artificial neural network (ANN)
# 

# In[71]:


mlp_lang = combined_df_ML.copy()
# mlp_lang.drop(columns=['AM_S4', 'AM_S3'])

# To convert all these data into numerical value for decision tree
# Categorical data to scoring system
mlp_lang['Gender'] = getGender(combined_df_ML)
mlp_lang['Sec4_CardingLevel'] = getCarding(combined_df_ML)
mlp_lang['Course'] = getcourse_tier(combined_df_ML)
mlp_lang['ResidentialType'] = gethousing(combined_df_ML)
# Numerical data to scoring system
mlp_lang['MT'] = encode_psle(combined_df_ML, 'MT')
# mlp_lang['HMT'] = getHMT_psle(combined_df_ML)
# mlp_lang = mlp_lang[mlp_lang['ML_S4'].notna()]
# mlp_lang = mlp_lang[mlp_lang['ML_S3'].notna()]
# mlp_lang = mlp_lang[mlp_lang['ML_S2'].notna()]
# mlp_lang = mlp_lang[mlp_lang['ML_S1'].notna()]
# Columns that are applying KNN Imputer
cols = mlp_lang.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
mlp_lang = KNN_Imputer(mlp_lang, cols)
mlp_lang = pd.DataFrame(mlp_lang, columns=cols)

# Validation
print(mlp_lang.shape)


# In[72]:


# to ignore if want to include express and NA
mlp_math = mlp_math[mlp_math['Course'] == 1]
print(mlp_math.shape)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# scaler.fit_transform()  # scales data
# scaler.inverse_transform()  # transforms scaled data back into original form
X = mlp_lang[['TScore', 'Gender', 'Sec4_CardingLevel',
              'Course', 'MT', 'ResidentialType', 'ML_S3', 'ML_S2', 'ML_S1']]
y = mlp_lang[['ML_S4']]
X = X.to_numpy()
y = y.to_numpy()

X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=2)


# In[ ]:


from sklearn.neural_network import MLPRegressor
clf = MLPRegressor(alpha=0.1, hidden_layer_sizes=(100, 50, 30), max_iter=500,
                   activation='relu', solver='adam', random_state=2)
model = clf.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_test)
y_pred = y_pred.reshape(-1, 1)
preds = scaler.inverse_transform(y_pred).flatten().round()


# In[ ]:


y_test_final = []
for i in scaler.inverse_transform(y_test):
    y_test_final.append(getresults_arr(i))


# In[ ]:


mlp_preds = leeway(preds, 5)


# In[ ]:


correct = 0
for i in range(len(preds)):
    if y_test_final[i] in mlp_preds[i]:
        correct += 1
print(correct / len(preds))


# In[ ]:




# # BLSTM
# 
# ---
# 
#     Recurrence Neural Network
# 
# A deep learning model, Bi-directional long short-term memory networks are the most effecient way to do the task of predicting a student's grade. The recurrent neural networks gave greater accuracy as compared to feed forward neural networks, as they have memory and take into consideration the consistency of the student performance. To predict the accuracy of the model, a mathematical method-root mean square error was used.
# 

# In[ ]:


# from sklearn.preprocessing import MinMaxScaler

# # Supress warning for SettingWithCopyError
# pd.options.mode.chained_assignment = None

# # Library to rescale variables into the range [0,1]
# scaler = MinMaxScaler()

# # Temp to do the switch between languages
# # var = combined_df_HCL
# # var = combined_df_CL
# var = combined_df_ML

# # Dataset
# df_combined_BLSTM = var.copy()

# # Columns that are applying KNN Imputer
# cols = df_combined_BLSTM.columns.to_numpy()

# # Numerical data to scoring system
# df_combined_BLSTM['MT'] = encode_psle(df_combined_BLSTM, 'MT')
# if 'HMT' in cols:
#     df_combined_BLSTM['HMT'] = getHMT_psle(df_combined_BLSTM)
#     df_combined_BLSTM['HMT'] = pd.to_numeric(df_combined_BLSTM['HMT'])

# # Convert all to numeric
# df_combined_BLSTM['Gender'] = getGender(df_combined_BLSTM)
# df_combined_BLSTM['Sec4_CardingLevel'] = getCarding(df_combined_BLSTM)
# df_combined_BLSTM['Course'] = getcourse_tier(df_combined_BLSTM)
# df_combined_BLSTM['ResidentialType'] = gethousing(df_combined_BLSTM)
# # Convert from boolean to float
# df_combined_BLSTM['Sec4_CardingLevel'] = df_combined_BLSTM['Sec4_CardingLevel'].astype(int)

# # Grades with banding
# df_combined_BLSTM['ML_S4'] = getresults(df_combined_BLSTM, 'ML_S4')
# df_combined_BLSTM['ML_S3'] = getresults(df_combined_BLSTM, 'ML_S3')
# df_combined_BLSTM['ML_S2'] = getresults(df_combined_BLSTM, 'ML_S2')
# df_combined_BLSTM['ML_S1'] = getresults(df_combined_BLSTM, 'ML_S1')

# # Fill up the NaN based on KNN Imputer
# df_combined_BLSTM = KNN_Imputer(df_combined_BLSTM, cols)
# df_combined_BLSTM = pd.DataFrame(df_combined_BLSTM, columns=cols)

# # Remove serial number column as it is useless
# df_combined_BLSTM.drop(columns=['Serial number'], inplace=True)

# # Extract Response and Predictors
# X = df_combined_BLSTM.drop(columns=['ML_S4'])
# y = df_combined_BLSTM['ML_S4']

# # Scale continuous variable in the dataset
# X = scaler.fit_transform(X)

# # X: Dataframe; y: Series
# y = pd.Series(y)


# # Split the Dataset into random Train and Test
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=2)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# # Reshaping for nodes overall
# X_train = np.array(
#     X_train.reshape(-1, int(X_train.shape[0]), int(X_train.shape[1])))
# X_test = np.array(
#     X_test.reshape(-1, int(X_test.shape[0]), int(X_test.shape[1])))
# y_train = np.array(y_train.values.reshape(-1, int(y_train.shape[0]), 1))
# y_test = np.array(y_test.values.reshape(-1, int(y_test.shape[0]), 1))

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

# Supress warning for SettingWithCopyError
pd.options.mode.chained_assignment = None

# Library to rescale variables into the range [0,1]
scaler = MinMaxScaler()

# Temp to do the switch between languages
var = df_P6_HCL
# var = df_P6_CL
# var = df_P6_ML

# Dataset
df_psych_BLSTM = var.copy()
df_psych_BLSTM

# Columns that are applying KNN Imputer
cols = df_psych_BLSTM.columns.to_numpy()

# Numerical data to scoring system
df_psych_BLSTM['MT'] = encode_psle(df_psych_BLSTM, 'MT')
if 'HMT' in cols:
    df_psych_BLSTM['HMT'] = getHMT_psle(df_psych_BLSTM)
    df_psych_BLSTM['HMT'] = pd.to_numeric(df_psych_BLSTM['HMT'])

# Convert all to numeric
df_psych_BLSTM['Gender'] = getGender(df_psych_BLSTM)
df_psych_BLSTM['Sec4_CardingLevel'] = getCarding(df_psych_BLSTM)
df_psych_BLSTM['Course'] = getcourse_tier(df_psych_BLSTM)
df_psych_BLSTM['ResidentialType'] = gethousing(df_psych_BLSTM)
# Convert from boolean to float
df_psych_BLSTM['Sec4_CardingLevel'] = df_psych_BLSTM['Sec4_CardingLevel'].astype(int)

# Grades with banding
df_psych_BLSTM['HCL_S4'] = getresults(df_psych_BLSTM, 'HCL_S4')
df_psych_BLSTM['HCL_S3'] = getresults(df_psych_BLSTM, 'HCL_S3')
df_psych_BLSTM['HCL_S2'] = getresults(df_psych_BLSTM, 'HCL_S2')
df_psych_BLSTM['HCL_S1'] = getresults(df_psych_BLSTM, 'HCL_S1')

# Fill up the NaN based on KNN Imputer
df_psych_BLSTM = KNN_Imputer(df_psych_BLSTM, cols)
df_psych_BLSTM = pd.DataFrame(df_psych_BLSTM, columns=cols)

# Remove serial number column as it is useless
df_psych_BLSTM.drop(columns=['Serial number'], inplace=True)

# Extract Response and Predictors
X = df_psych_BLSTM.drop(columns=['HCL_S4'])
y = df_psych_BLSTM['HCL_S4']

# Scale continuous variable in the dataset
X = scaler.fit_transform(X)

# X: Dataframe; y: Series
y = pd.Series(y)

# Split the Dataset into random Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Reshaping for nodes overall
X_train = np.array(
    X_train.reshape(-1, int(X_train.shape[0]), int(X_train.shape[1])))
X_test = np.array(
    X_test.reshape(-1, int(X_test.shape[0]), int(X_test.shape[1])))
y_train = np.array(y_train.values.reshape(-1, int(y_train.shape[0]), 1))
y_test = np.array(y_test.values.reshape(-1, int(y_test.shape[0]), 1))

# In[ ]:


# Feature Scaling
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout, Bidirectional

regressor = Sequential()
regressor.add(LSTM(units=20, activation='sigmoid',
              input_shape=(None, 51), return_sequences=True))  # Tweak the parameter to match the column size
regressor.add(Bidirectional(
    LSTM(units=20, activation='sigmoid', return_sequences=True)))
# regressor.add(Dropout(0.20))
regressor.add(Bidirectional(
    LSTM(units=20, activation='sigmoid', return_sequences=True)))
regressor.add(Bidirectional(
    LSTM(units=20, activation='sigmoid', return_sequences=True)))
# regressor.add(Bidirectional(LSTM(units=20, activation='sigmoid')))
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(X_train, y_train, batch_size=32, epochs=400)

# Based on our Trained Model
predicted = regressor.predict(X_test)

# Compare real values to predicted
# 

# In[ ]:


predicted = predicted.ravel()
y_test = y_test.ravel()

d = {'Actual Result': y_test, 'Predicted by Model': predicted}
df = pd.DataFrame(data=d)
df.round(2)[:10]  # Show 10 results


# In[ ]:


# Goodness of Fit of the DDN Model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print("Mean Squared Error: ", mean_squared_error(y_test, predicted))
print("Mean Absolute Error: ", mean_absolute_error(y_test, predicted))
print("R^2 Error: ", r2_score(y_test, predicted))


# In[ ]:


correct = 0
for i in range(len(predicted)):
    if abs(y_test[i] - predicted[i].round()) <= 0:
        correct += 1
print(correct / len(predicted))



# Comment: Mean absolute value shows that on average, the model makes a mistake of around 1 band. This is considered as a poor result. To improve the model, consider inputing more data. I believe BLSTM is a reliable guage with larger data
# 

# In[ ]:


RMSE = mean_squared_error(predicted, y_test)**0.5
c1 = 0
c2 = 0
c3 = 0
for i in range(len(predicted)):
    err = (abs(predicted[i] - y_test[i])*100)/y_test[i]
    if err <= 5:
        c1 = c1+1
    elif err <= 10:
        c2 = c2+1
    elif err <= 20:
        c3 = c3+1
print("no of student marks predicted with less than or equal to 5% error", c1)
print("no of student marks predicted with less than or equal to 10% error", c2)
print("no of student marks predicted with less than or equal to 20% error", c3)


'''compare y_test and predicted to get an underdtand the prediction'''


# In[ ]:




# In[ ]:




# # XGBOOST banding
# 

# In[ ]:


from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df_filled = combined_df.copy()

# To convert all these data into numerical value for decision tree
# Categorical data to scoring system
df_filled['Gender'] = getGender(combined_df)
df_filled['Sec4_CardingLevel'] = getCarding(combined_df)
df_filled['Course'] = getcourse_tier(combined_df)
df_filled['ResidentialType'] = gethousing(combined_df)
# Numerical data to scoring system
df_filled['EL'] = encode_psle(combined_df, 'EL')
df_filled['English_S4'] = getresults(combined_df, 'English_S4')
df_filled = df_filled[df_filled['English_S4'].notna()]
# Columns that are applying KNN Imputer
cols = df_filled.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
df_filled = KNN_Imputer(df_filled, cols)
df_filled = pd.DataFrame(df_filled, columns=cols)

# Validation
print(df_filled.shape)


# In[ ]:


# to ignore if fitting model with both express and NA
df_filled = df_filled[df_filled['Course'] == 1]


# In[ ]:


# scaler.fit_transform()  # scales data
# scaler.inverse_transform()  # transforms scaled data back into original form
scaler = MinMaxScaler()
X = df_filled[['TScore', 'Gender', 'Sec4_CardingLevel',
               'Course', 'EL', 'ResidentialType']]
y = df_filled[['English_S4']]
X = X.to_numpy()
y = y.to_numpy()

X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=2)
y_test_final = scaler.inverse_transform(y_test)

xgr = XGBRegressor()
xgr.fit(X_train, y_train)


# In[ ]:


# Predictions from model
xg_preds = xgr.predict(X_test)
xg_preds = xg_preds.reshape(-1, 1)
xg_preds = scaler.inverse_transform(xg_preds).flatten().round()


# In[ ]:


len(xg_preds)


# In[ ]:


correct = 0
for i in range(len(xg_preds)):
    # allowance of 1 increases prediction accuracy
    if abs(xg_preds[i] - y_test_final[i]) <= 1:
        correct += 1
print(correct / len(xg_preds))


# # XGBoost Regressor
# 
# - predicts numeric values
# 

# In[ ]:


xg_lang = combined_df_ML.copy()
# xg_lang.drop(columns = ['EM_S4'])

# To convert all these data into numerical value for decision tree
# Categorical data to scoring system
xg_lang['Gender'] = getGender(combined_df_ML)
xg_lang['Sec4_CardingLevel'] = getCarding(combined_df_ML)
xg_lang['Course'] = getcourse_tier(combined_df_ML)
xg_lang['ResidentialType'] = gethousing(combined_df_ML)
# Numerical data to scoring system
xg_lang['MT'] = encode_psle(combined_df_ML, 'MT')
# xg_lang['HMT'] = getHMT_psle(combined_df_ML)
# xg_lang = xg_lang[xg_lang['ML_S4'].notna()]
# xg_lang = xg_lang[xg_lang['ML_S3'].notna()]
# xg_lang = xg_lang[xg_lang['ML_S2'].notna()]
# xg_lang = xg_lang[xg_lang['ML_S1'].notna()]
# Columns that are applying KNN Imputer
cols = xg_lang.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
xg_lang = KNN_Imputer(xg_lang, cols)
xg_lang = pd.DataFrame(xg_lang, columns=cols)

# Validation
print(xg_lang.shape)


# In[ ]:


# to ignore if want to include express and NA
xg_math = xg_math[xg_math['Course'] == 1]
print(xg_math.shape)


# In[ ]:


from xgboost import XGBRegressor
scaler = MinMaxScaler()
# scaler.fit_transform()  # scales data
# scaler.inverse_transform()  # transforms scaled data back into original form
X = xg_lang[['TScore', 'Gender', 'Sec4_CardingLevel',
             'Course', 'MT', 'ResidentialType', 'ML_S3', 'ML_S2', 'ML_S1']]
y = xg_lang[['ML_S4']]
X = X.to_numpy()
y = y.to_numpy()

X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=2)
y_test_final = scaler.inverse_transform(y_test).flatten()

xgr = XGBRegressor(learning_rate=0.2, max_depth=4)
#xgr = XGBRegressor()
xgr.fit(X_train, y_train)


# In[ ]:


# converting test results into grades
y_test_grade = []
for i in y_test_final:
    y_test_grade.append(getresults_arr(i))


# In[ ]:


# Predictions from model
xg_preds_cont = xgr.predict(X_test)
xg_preds_cont = xg_preds_cont.reshape(-1, 1)
xg_preds_cont = scaler.inverse_transform(xg_preds_cont).flatten().round()
xg_preds_cont


# In[ ]:


xg_preds_rg = leeway(xg_preds_cont, 3)


# In[ ]:


correct = 0
for i in range(len(xg_preds_cont)):
    if y_test_grade[i] in xg_preds_rg[i]:
        correct += 1
print(correct / len(xg_preds_cont))


# # XGBoost with P6 psychological data
# 

# In[ ]:


# df_test = pd.concat([df_P6_with_S4_2020, df_P6_with_S4_2021])
# df_test = df_test[['TScore', 'Gender', 'Sec4_CardingLevel', 'Course', 'Maths', 'ResidentialType', 'AM_S4', 'AM_S3', 'EM_S4', 'EM_S3', 'EM_S2', 'EM_S1',
#                                        'Percentage (%)', 'Percentage (%).1', 'Percentage (%).2', 'Percentage (%).3', 'Percentage (%).4', 'Percentage (%).5',
#                                        'Q1 M', 'Q1F', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8a', 'Q8b', 'Q8c', 'Q1.6', 'Q2a', 'Q2b', 'Q2c', 'Q2d', 'Q2e', 'Q2f',
#                                        'Q2g', 'Q2h', 'Q2i', 'Q2j', 'Q2k', 'Q3.7', 'Q4a', 'Q4b', 'Q4c', 'Q4d',
#                                        'Q4e', 'Q4f', 'Q4g', 'Q4h', 'Q4i', 'Q4j', 'Q4k']]


# In[ ]:


df_test1 = df_P6_CL.copy()
df_test1['Gender'] = getGender(df_P6_CL)
df_test1['Sec4_CardingLevel'] = getCarding(df_P6_CL)
df_test1['Course'] = getcourse_tier(df_P6_CL)
df_test1['ResidentialType'] = gethousing(df_P6_CL)

# Numerical data to scoring system
df_test1['MT'] = encode_psle(df_P6_CL, 'MT')
# df_test1['HMT'] = getHMT_psle(df_P6_CL)
# df_test1 = df_test1[df_test1['CL_S4'].notna()]
# df_test1 = df_test1[df_test1['CL_S3'].notna()]
# df_test1 = df_test1[df_test1['CL_S2'].notna()]
# df_test1 = df_test1[df_test1['CL_S1'].notna()]

# Columns that are applying KNN Imputer
cols = df_test1.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
df_test1 = KNN_Imputer(df_test1, cols)
df_test1 = pd.DataFrame(df_test1, columns=cols)

# Reset the df index
df_test1.reset_index(inplace=True)
print(df_test1.shape)
df_test1.head(5)


# In[ ]:


# to ignore if want to include express and NA
df_test1 = df_test1[df_test1['Course'] == 1]
print(df_test1.shape)


# In[ ]:


from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler()

X = df_test1.copy()
print(X.shape)
X.drop(columns=['CL_S4'], inplace=True)
y = df_test1[['CL_S4']]

X = X.to_numpy()
y = y.to_numpy()

X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=2)
y_test_final = scaler.inverse_transform(y_test).flatten()


# In[ ]:


xgr = XGBRegressor()
xgr.fit(X_train, y_train)


# In[ ]:


# converting test results into grades
y_test_grade = []
for i in y_test_final:
    y_test_grade.append(getresults_arr(i))


# In[ ]:


# Predictions from model
xg_p6preds = xgr.predict(X_test)
xg_p6preds = xg_p6preds.reshape(-1, 1)
xg_p6preds = scaler.inverse_transform(xg_p6preds).flatten().round()


# In[ ]:


xg_p6preds_rg = leeway(xg_p6preds, 5)


# In[ ]:


correct = 0
for i in range(len(xg_p6preds)):
    if y_test_grade[i] in xg_p6preds_rg[i]:
        correct += 1
        # print(y_test_grade[i], xg_p6preds_rg[i])
print(correct / len(xg_p6preds))
# W/o S1 to S3 El: 0.53125


# # MLP P6
# 

# In[ ]:


df_test1 = df_P6_ML.copy()
df_test1['Gender'] = getGender(df_P6_ML)
df_test1['Sec4_CardingLevel'] = getCarding(df_P6_ML)
df_test1['Course'] = getcourse_tier(df_P6_ML)
df_test1['ResidentialType'] = gethousing(df_P6_ML)

# Numerical data to scoring system
df_test1['MT'] = encode_psle(df_P6_ML, 'MT')
# df_test1['HMT'] = getHMT_psle(df_P6_ML)
df_test1 = df_test1[df_test1['ML_S4'].notna()]
df_test1 = df_test1[df_test1['ML_S3'].notna()]
df_test1 = df_test1[df_test1['ML_S2'].notna()]
df_test1 = df_test1[df_test1['ML_S1'].notna()]

# Columns that are applying KNN Imputer
cols = df_test1.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academay'))

# Fill up the NaN based on KNN Imputer
df_test1 = KNN_Imputer(df_test1, cols)
df_test1 = pd.DataFrame(df_test1, columns=cols)

# Reset the df index
df_test1.reset_index(inplace=True)
print(df_test1.shape)
df_test1.head(5)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler()

X = df_test1.copy()
print(X.shape)
X.drop(columns=['ML_S4'], inplace=True)
# for col in X.columns:
#     print(col)
y = df_test1[['ML_S4']]

X = X.to_numpy()
y = y.to_numpy()

X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=2)
y_test_final = scaler.inverse_transform(y_test).flatten()


# In[ ]:


from sklearn.neural_network import MLPRegressor
clf = MLPRegressor(alpha=0.01, hidden_layer_sizes=(100, 50, 30), max_iter=1000,
                   activation='relu', solver='adam', random_state=2)
model = clf.fit(X_train, y_train)


# In[ ]:


# converting test results into grades
y_test_grade = []
for i in y_test_final:
    y_test_grade.append(getresults_arr(i))


# In[ ]:


# Predictions from model
mlp_p6preds = model.predict(X_test)
mlp_p6preds = mlp_p6preds.reshape(-1, 1)
mlp_p6preds = scaler.inverse_transform(mlp_p6preds).flatten().round()


# In[ ]:


mlp_p6preds_rg = leeway(mlp_p6preds, 5)


# In[ ]:


correct = 0
for i in range(len(mlp_p6preds)):
    if y_test_grade[i] in mlp_p6preds_rg[i]:
        correct += 1
        # print(y_test_grade[i], mlp_p6preds_rg[i])
print(correct / len(mlp_p6preds))

