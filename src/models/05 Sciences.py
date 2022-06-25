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

# ### Sorted in accordance to Year 4, 3, 2, 1
# 
#     DATAFRAME 2021
#     'Sci'
# 
#     '2021 Results', 'Biology', 'Chemistry', 'Physics', 'Science (Chemistry)', 'Science (Physics)', 'Science (Biology)', 'Science(Physics/Chemistry)', 'Science(Biology/Chemistry)', 'Science (Physics/Chemistry)', 'Science (Biology/Chemistry)', 'Science',
# 
#     '2020 Result',
#     'Biology.1', 'Chemistry.1', 'Physics.1', 'Science.1', 'Science (Biology).1', 'Science (Chemistry).1', 'Science (Physics).1', 'Science(Biology/Chemistry).1', 'Science(Physics/Chemistry).1',
# 
#     '2019 Results',
#     'Science.2'
# 
#     '2018 Results',
#     'Science.3'
# 

# In[12]:


print(set(list(data_S4_2018)) - set(list(data_S4_2016)))


# In[13]:


def update_sci(df, ideal, other):
    data = df[ideal].combine_first(df[other])
    df[ideal] = data


# In[14]:


# Merge combined sci cols for dataframes
phy_chem = ['Science(Physics/Chemistry)', 'Science (Physics/Chemistry)']
bio_chem = ['Science(Biology/Chemistry)', 'Science (Biology/Chemistry)']
update_sci(data_S4_2019, phy_chem[0], phy_chem[1])
update_sci(data_S4_2020, phy_chem[0], phy_chem[1])
update_sci(data_S4_2021, phy_chem[0], phy_chem[1])


# In[15]:


# validation that it's been merged
data_S4_2020['Science(Physics/Chemistry)'][26]


# In[16]:


print(list(data_S4_2019))


# In[17]:


# Truncating dataframe to columns that we only need
lst21 = ['Serial number', 'Gender', 'Race', 'Academy', 'Sec4_CardingLevel', 'Course', 'TScore', 'ResidentialType',
         # S4 + S3 Science is Science (Biology/Physics)
         'Science', 'Science.1', 'Science.2', 'Science.3', 'Sci',
         'Science(Physics/Chemistry)', 'Science(Biology/Chemistry)',
         'Science(Physics/Chemistry).1', 'Science(Biology/Chemistry).1',
         'Science (Physics)', 'Science (Physics).1',
         'Science (Chemistry)', 'Science (Chemistry).1',
         'Science (Biology)', 'Science (Biology).1',
         'Biology', 'Biology.1', 'Physics', 'Physics.1',
         'Chemistry', 'Chemistry.1']

# List of the respective dataframe columns
lst20 = list(data_S4_2020)
lst19 = list(data_S4_2019)
lst18 = list(data_S4_2018)
lst17 = list(data_S4_2017)
lst16 = list(data_S4_2016)

# Check if master list has all science related col name
diff1 = (set(lst20) - set(lst21))
diff2 = (set(lst19) - set(lst21))
diff3 = (set(lst18) - set(lst21))
diff4 = (set(lst17) - set(lst21))
diff5 = (set(lst16) - set(lst21))
# Unified the set to remove duplicate columns for easy readiability
union = set.union(diff1, diff2, diff3, diff4, diff5)
# print(union) #Uncomment this to ensure that there is no science related columns

# # Intersection between 2021 identified columns to the respective columns
intersection21 = lst21
intersection20 = list(set(lst21).intersection(lst20))
intersection19 = list(set(lst21).intersection(lst19))
intersection18 = list(set(lst21).intersection(lst18))
intersection17 = list(set(lst21).intersection(lst17))
intersection16 = list(set(lst21).intersection(lst16))

# Standardize looking of dataframe
intersection21.sort()
intersection20.sort()
intersection19.sort()
intersection18.sort()
intersection17.sort()
intersection16.sort()


# In[18]:


# Truncating dataframe to columns that we only need
df_S4_2021 = pd.DataFrame(data_S4_2021[intersection21])
df_S4_2020 = pd.DataFrame(data_S4_2020[intersection20])
df_S4_2019 = pd.DataFrame(data_S4_2019[intersection19])
df_S4_2018 = pd.DataFrame(data_S4_2018[intersection18])
df_S4_2017 = pd.DataFrame(data_S4_2017[intersection17])
df_S4_2016 = pd.DataFrame(data_S4_2016[intersection16])


# In[19]:


# Renaming to avoid confusion
df_S4_2016 = df_S4_2016.rename(columns={'Biology': 'Bio_S4', 'Biology.1': 'Bio_S3', 'Chemistry': 'Chem_S4', 'Chemistry.1': 'Chem_S3', 'Physics': 'Phy_S4', 'Physics.1': 'Phy_S3',
                                        'Science': 'Sci_S2', 'Science.1': 'Sci_S1', 'Science (Biology)': 'Comb_Bio_S4', 'Science (Biology).1': 'Comb_Bio_S3',
                                        'Science (Chemistry)': 'Comb_Chem_S4', 'Science (Chemistry).1': 'Comb_Chem_S3', 'Science (Physics)': 'Comb_Phy_S4', 'Science (Physics).1': 'Comb_Phy_S3'
                                        })
df_S4_2017 = df_S4_2017.rename(columns={'Biology': 'Bio_S4', 'Biology.1': 'Bio_S3', 'Chemistry': 'Chem_S4', 'Chemistry.1': 'Chem_S3', 'Physics': 'Phy_S4', 'Physics.1': 'Phy_S3',
                                        'Science': 'Sci_S2', 'Science.1': 'Sci_S1', 'Science (Biology)': 'Comb_Bio_S4', 'Science (Biology).1': 'Comb_Bio_S3',
                                        'Science (Chemistry)': 'Comb_Chem_S4', 'Science (Chemistry).1': 'Comb_Chem_S3', 'Science (Physics)': 'Comb_Phy_S4', 'Science (Physics).1': 'Comb_Phy_S3'
                                        })
df_S4_2018 = df_S4_2018.rename(columns={'Biology': 'Bio_S4', 'Biology.1': 'Bio_S3', 'Chemistry': 'Chem_S4', 'Chemistry.1': 'Chem_S3', 'Physics': 'Phy_S4', 'Physics.1': 'Phy_S3',
                                        'Science': 'Sci_S2', 'Science.1': 'Sci_S1', 'Science (Biology)': 'Comb_Bio_S4', 'Science (Biology).1': 'Comb_Bio_S3',
                                        'Science (Chemistry)': 'Comb_Chem_S4', 'Science (Chemistry).1': 'Comb_Chem_S3', 'Science (Physics)': 'Comb_Phy_S4', 'Science (Physics).1': 'Comb_Phy_S3'
                                        })
# 2019 doesn't have combined science cols for S3, need to create
# Science is known to be combined biology + physics
df_S4_2019 = df_S4_2019.rename(columns={'Biology': 'Bio_S4', 'Biology.1': 'Bio_S3', 'Chemistry': 'Chem_S4', 'Chemistry.1': 'Chem_S3', 'Physics': 'Phy_S4', 'Physics.1': 'Phy_S3',
                                        'Science.1': 'Sci_S2', 'Science.2': 'Sci_S1', 'Science (Biology)': 'Comb_Bio_S4', 'Science (Biology).1': 'Comb_Bio_S3',
                                        'Science (Chemistry)': 'Comb_Chem_S4', 'Science (Chemistry).1': 'Comb_Chem_S3', 'Science (Physics)': 'Comb_Phy_S4', 'Science (Physics).1': 'Comb_Phy_S3',
                                        'Science': 'Comb_Sci(Phy/Bio)_S4', 'Science(Physics/Chemistry)': 'Comb_Sci(Phy/Chem)_S4', 'Science(Biology/Chemistry)': 'Comb_Sci(Bio/Chem)_S4'
                                        })
# 2020 and 2021 has combined science cols for S3 and S4
df_S4_2020 = df_S4_2020.rename(columns={'Biology': 'Bio_S4', 'Biology.1': 'Bio_S3', 'Chemistry': 'Chem_S4', 'Chemistry.1': 'Chem_S3', 'Physics': 'Phy_S4', 'Physics.1': 'Phy_S3',
                                        'Science.2': 'Sci_S2', 'Science.3': 'Sci_S1', 'Science (Biology)': 'Comb_Bio_S4', 'Science (Biology).1': 'Comb_Bio_S3',
                                        'Science (Chemistry)': 'Comb_Chem_S4', 'Science (Chemistry).1': 'Comb_Chem_S3', 'Science (Physics)': 'Comb_Phy_S4', 'Science (Physics).1': 'Comb_Phy_S3',
                                        'Science': 'Comb_Sci(Phy/Bio)_S4', 'Science.1': 'Comb_Sci(Phy/Bio)_S3',
                                        'Science(Physics/Chemistry)': 'Comb_Sci(Phy/Chem)_S4', 'Science(Physics/Chemistry).1': 'Comb_Sci(Phy/Chem)_S3',
                                        'Science(Biology/Chemistry)': 'Comb_Sci(Bio/Chem)_S4', 'Science(Biology/Chemistry).1': 'Comb_Sci(Bio/Chem)_S3'
                                        })
# No difference
df_S4_2021 = df_S4_2021.rename(columns={'Biology': 'Bio_S4', 'Biology.1': 'Bio_S3', 'Chemistry': 'Chem_S4', 'Chemistry.1': 'Chem_S3', 'Physics': 'Phy_S4', 'Physics.1': 'Phy_S3',
                                        'Science.2': 'Sci_S2', 'Science.3': 'Sci_S1', 'Science (Biology)': 'Comb_Bio_S4', 'Science (Biology).1': 'Comb_Bio_S3',
                                        'Science (Chemistry)': 'Comb_Chem_S4', 'Science (Chemistry).1': 'Comb_Chem_S3', 'Science (Physics)': 'Comb_Phy_S4', 'Science (Physics).1': 'Comb_Phy_S3',
                                        'Science': 'Comb_Sci(Phy/Bio)_S4', 'Science.1': 'Comb_Sci(Phy/Bio)_S3',
                                        'Science(Physics/Chemistry)': 'Comb_Sci(Phy/Chem)_S4', 'Science(Physics/Chemistry).1': 'Comb_Sci(Phy/Chem)_S3',
                                        'Science(Biology/Chemistry)': 'Comb_Sci(Bio/Chem)_S4', 'Science(Biology/Chemistry).1': 'Comb_Sci(Bio/Chem)_S3'
                                        })


#     2016 --> S3 combined humanities (No name)
#     2017 --> Dun even have combined humanities result
#     2018 --> Dun even have combined humanities result
#     2019 --> S4 Combined Humanities(Econs/Geo) (Econs/His) (SS/Geo)	(SS/His)
#     2020 --> S4 Combined Humanities(Econs/Geo) (Econs/His) (SS/Geo)	(SS/His) + S3 (SS/His)', 'S3 (SS/Geo)'
#     2020 --> S4 Combined Humanities(Econs/Geo) (Econs/His) (SS/Geo)	(SS/His) + S3 (SS/His)', 'S3 (SS/Geo)'
# 

# In[20]:


print(list(df_S4_2016))
print()
print(list(df_S4_2017))
print()
print(list(df_S4_2018))
print()
print(list(df_S4_2019))
print()
print(list(df_S4_2020))
print()
print(list(df_S4_2021))


# In[21]:


dfs = [df_S4_2016, df_S4_2017, df_S4_2018, df_S4_2019, df_S4_2020, df_S4_2021]

numeric_col = ['Bio_S4', 'Bio_S3', 'Chem_S4', 'Chem_S3', 'Phy_S4', 'Phy_S3',
               'Comb_Sci(Phy/Bio)_S4', 'Comb_Bio_S4', 'Comb_Bio_S3', 'Comb_Chem_S4',
               'Comb_Chem_S3', 'Comb_Phy_S4', 'Comb_Phy_S3', 'Comb_Sci(Bio/Chem)_S4',
               'Comb_Sci(Bio/Chem)_S3', 'Comb_Sci(Phy/Chem)_S4', 'Comb_Sci(Phy/Chem)_S3',
               'Comb_Sci(Phy/Bio)_S3', 'Sci_S2', 'Sci_S1']

for df in dfs:  # For loop thru the df
    for col in df.columns:
        # Standardize nan
        df[col] = df[col].replace({'-': np.nan, '0': np.nan, 0: np.nan})
        # Turn columns into numeric values for arithmetic operation
        if col in numeric_col:
            df[col] = pd.to_numeric(df[col])


# In[22]:


dfs = [df_S4_2016, df_S4_2017, df_S4_2018]

# Create combined science columns
for df in dfs:
    df['Comb_Sci(Bio/Chem)_S4'] = 0.5 * (df['Comb_Bio_S4'] +
                                         df['Comb_Chem_S4'])  # 'Comb_Sci(Bio/Chem)_S4'
    df['Comb_Sci(Bio/Chem)_S3'] = 0.5 * (df['Comb_Bio_S3'] +
                                         df['Comb_Chem_S3'])  # 'Comb_Sci(Bio/Chem)_S3'
    df['Comb_Sci(Phy/Chem)_S4'] = 0.5 * (df['Comb_Phy_S4'] +
                                         df['Comb_Chem_S4'])  # 'Comb_Sci(Phy/Chem)_S4'
    df['Comb_Sci(Phy/Chem)_S3'] = 0.5 * (df['Comb_Phy_S3'] +
                                         df['Comb_Chem_S3'])  # 'Comb_Sci(Phy/Chem)_S3'
    df['Comb_Sci(Phy/Bio)_S4'] = 0.5 * (df['Comb_Phy_S4'] +
                                        df['Comb_Bio_S4'])  # 'Comb_Sci(Phy/Bio)_S4'
    df['Comb_Sci(Phy/Bio)_S3'] = 0.5 * (df['Comb_Phy_S3'] +
                                        df['Comb_Bio_S3'])  # 'Comb_Sci(Phy/Bio)_S3'

# 2019 Combined Science S3
df_S4_2019['Comb_Sci(Bio/Chem)_S3'] = 0.5 * (df['Comb_Bio_S3'] +
                                             df['Comb_Chem_S3'])  # 'Comb_Sci(Bio/Chem)_S3'
df_S4_2019['Comb_Sci(Phy/Chem)_S3'] = 0.5 * (df['Comb_Phy_S3'] +
                                             df['Comb_Chem_S3'])  # 'Comb_Sci(Phy/Chem)_S3'
df_S4_2019['Comb_Sci(Phy/Bio)_S3'] = 0.5 * (df['Comb_Bio_S3'] +
                                            df['Comb_Phy_S3'])  # 'Comb_Sci(Phy/Bio)_S3'

# Ensure 2019 onwards coombined sciences are filled
dfs = [df_S4_2019, df_S4_2020, df_S4_2021]
for df in dfs:
    # Sec 4
    df.loc[df['Comb_Sci(Bio/Chem)_S4'].isna(), ['Comb_Sci(Bio/Chem)_S4']
           ] = 0.5 * (df['Comb_Bio_S4'] + df['Comb_Chem_S4'])
    df.loc[df['Comb_Sci(Phy/Chem)_S4'].isna(), ['Comb_Sci(Phy/Chem)_S4']
           ] = 0.5 * (df['Comb_Phy_S4'] + df['Comb_Chem_S4'])
    df.loc[df['Comb_Sci(Phy/Bio)_S4'].isna(), ['Comb_Sci(Phy/Bio)_S4']
           ] = 0.5 * (df['Comb_Bio_S4'] + df['Comb_Phy_S4'])
    # Sec 3
    df.loc[df['Comb_Sci(Bio/Chem)_S3'].isna(), ['Comb_Sci(Bio/Chem)_S3']
           ] = 0.5 * (df['Comb_Bio_S3'] + df['Comb_Chem_S3'])
    df.loc[df['Comb_Sci(Phy/Chem)_S3'].isna(), ['Comb_Sci(Phy/Chem)_S3']
           ] = 0.5 * (df['Comb_Phy_S3'] + df['Comb_Chem_S3'])
    df.loc[df['Comb_Sci(Phy/Bio)_S3'].isna(), ['Comb_Sci(Phy/Bio)_S3']
           ] = 0.5 * (df['Comb_Bio_S3'] + df['Comb_Phy_S3'])


# In[23]:


# Merging all dfs together (Row order is preserved)
dfs = [df_S4_2016, df_S4_2017, df_S4_2018, df_S4_2019, df_S4_2020, df_S4_2021]
combined_df = pd.concat(dfs)

# Setting boolean for carded athlete
col = 'Sec4_CardingLevel'
combined_df[col] = combined_df[col].replace(np.nan, 0)
combined_df[col] = combined_df[col].replace(['L3', 'Y', 'L4P', 'L4', 'YT',
                                             'TL3', 'E3', 'B4', 'ET3', 'Y+'], 1)
combined_df[col] = pd.to_numeric(combined_df[col])

# Drop Academy and Race for better accuracy
combined_df.drop(columns=['Academy', 'Race'], inplace=True)
combined_df = combined_df.reset_index(drop=True)


# In[24]:


# for reference to the cols
print(list(combined_df))


# In[25]:


# Split the dataset into respective df for analysis

# Pure Bio data
pure_bio = combined_df[['Course', 'Sci', 'Sci_S1', 'Sci_S2', 'Bio_S4', 'Bio_S3',
                        'Gender', 'Sec4_CardingLevel', 'Serial number', 'TScore', 'ResidentialType']]

# Pure Chem data
pure_chem = combined_df[['Course', 'Sci', 'Sci_S1', 'Sci_S2', 'Chem_S4', 'Chem_S3',
                        'Gender', 'Sec4_CardingLevel', 'Serial number', 'TScore', 'ResidentialType']]

# Pure Phy data
pure_phy = combined_df[['Course', 'Sci', 'Sci_S1', 'Sci_S2', 'Phy_S4', 'Phy_S3',
                        'Gender', 'Sec4_CardingLevel', 'Serial number', 'TScore', 'ResidentialType']]

# Combined Phy/Chem
combined_phy_chem = combined_df[['Course', 'Sci', 'Sci_S1', 'Sci_S2', 'Comb_Sci(Phy/Chem)_S4', 'Comb_Sci(Phy/Chem)_S3',
                                 'Gender', 'Sec4_CardingLevel', 'Serial number', 'TScore', 'ResidentialType']]

# Combined Bio/Chem
combined_bio_chem = combined_df[['Course', 'Sci', 'Sci_S1', 'Sci_S2', 'Comb_Sci(Bio/Chem)_S4', 'Comb_Sci(Bio/Chem)_S3',
                                 'Gender', 'Sec4_CardingLevel', 'Serial number', 'TScore', 'ResidentialType']]

# Combined Phy/Bio
combined_phy_bio = combined_df[['Course', 'Sci', 'Sci_S1', 'Sci_S2', 'Comb_Sci(Phy/Bio)_S4', 'Comb_Sci(Phy/Bio)_S3',
                                'Gender', 'Sec4_CardingLevel', 'Serial number', 'TScore', 'ResidentialType']]


# Drop row if all the key elements are missing
pure_bio = pure_bio.dropna(
    subset=['Bio_S4', 'Bio_S3'], how='any')  # Pure Geo
pure_chem = pure_chem.dropna(
    subset=['Chem_S4', 'Chem_S3'], how='any')  # Pure History
pure_phy = pure_phy.dropna(
    subset=['Phy_S4', 'Phy_S3'], how='any')  # Missing data
combined_phy_chem = combined_phy_chem.dropna(
    subset=['Comb_Sci(Phy/Chem)_S4', 'Comb_Sci(Phy/Chem)_S3'], how='any')  # Missing data
combined_bio_chem = combined_bio_chem.dropna(
    subset=['Comb_Sci(Bio/Chem)_S4', 'Comb_Sci(Bio/Chem)_S3'], how='any')  # Missing data
combined_phy_bio = combined_phy_bio.dropna(
    subset=['Comb_Sci(Phy/Bio)_S4', 'Comb_Sci(Phy/Bio)_S3'], how='any')  # Missing data

# Reset the index
pure_bio = pure_bio.reset_index(drop=True)
pure_chem = pure_chem.reset_index(drop=True)
pure_phy = pure_phy.reset_index(drop=True)
combined_phy_chem = combined_phy_chem.reset_index(drop=True)
combined_bio_chem = combined_bio_chem.reset_index(drop=True)
combined_phy_bio = combined_phy_bio.reset_index(drop=True)

print(pure_bio.shape, pure_chem.shape, pure_phy.shape,
      combined_phy_chem.shape, combined_bio_chem.shape, combined_phy_bio.shape)


# In[26]:


print(list(combined_phy_bio))


# ### DataSet (Combined)
# 
# | DataFrame         | Shape     | Cols                                                                                                                                                                 |
# | ----------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | pure_bio          | (89, 11)  | ['Course', 'Sci', 'Sci_S1', 'Sci_S2', 'Bio_S4', 'Bio_S3', 'Gender', 'Sec4_CardingLevel', 'Serial number', 'TScore', 'ResidentialType']                               |
# | pure_chem         | (174, 11) | ['Course', 'Sci', 'Sci_S1', 'Sci_S2', 'Chem_S4', 'Chem_S3', 'Gender', 'Sec4_CardingLevel', 'Serial number', 'TScore', 'ResidentialType']                             |
# | pure_phy          | (85, 11)  | ['Course', 'Sci', 'Sci_S1', 'Sci_S2', 'Phy_S4', 'Phy_S3', 'Gender', 'Sec4_CardingLevel', 'Serial number', 'TScore', 'ResidentialType']                               |
# | combined_phy_chem | (165, 11) | ['Course', 'Sci', 'Sci_S1', 'Sci_S2', 'Comb_Sci(Phy/Chem)_S4', 'Comb_Sci(Phy/Chem)_S3', 'Gender', 'Sec4_CardingLevel', 'Serial number', 'TScore', 'ResidentialType'] |
# | combined_bio_chem | (122, 11) | ['Course', 'Sci', 'Sci_S1', 'Sci_S2', 'Comb_Sci(Bio/Chem)_S4', 'Comb_Sci(Bio/Chem)_S3', 'Gender', 'Sec4_CardingLevel', 'Serial number', 'TScore', 'ResidentialType'] |
# | combined_phy_bio  | (12, 11)  | ['Course', 'Sci', 'Sci_S1', 'Sci_S2', 'Comb_Sci(Phy/Bio)_S4', 'Comb_Sci(Phy/Bio)_S3', 'Gender', 'Sec4_CardingLevel', 'Serial number', 'TScore', 'ResidentialType']   |
# 

# ## DataFrame:
# 
#     Merging P6 Screening Result with Secondary School Result
#     P6 2016 + Secondary 2020
#     P6 2017 + Secondary 2021
# 

# In[27]:


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


# In[28]:


# Convert the column to type float so that we can merge
data_P6_2016['Serial number'] = data_P6_2016['Serial number'].astype(int)
data_P6_2017['Serial number'] = data_P6_2017['Serial number'].astype(int)

# Combined both data
# Merge the dataframe
df_psych_original = pd.concat([data_P6_2016, data_P6_2017])

# Merging of data (Paring of S/No. is unique)
df_P6_pure_bio = df_psych_original.merge(
    pure_bio, how='inner', on=['Serial number'])
df_P6_pure_chem = df_psych_original.merge(
    pure_chem, how='inner', on=['Serial number'])
df_P6_pure_phy = df_psych_original.merge(
    pure_phy, how='inner', on=['Serial number'])
df_P6_combined_phy_chem = df_psych_original.merge(
    combined_phy_chem, how='inner', on=['Serial number'])
df_P6_combined_bio_chem = df_psych_original.merge(
    combined_bio_chem, how='inner', on=['Serial number'])
df_P6_combined_phy_bio = df_psych_original.merge(
    combined_phy_bio, how='inner', on=['Serial number'])

# Reset index
df_P6_pure_bio = df_P6_pure_bio.reset_index(drop=True)
df_P6_pure_chem = df_P6_pure_chem.reset_index(drop=True)
df_P6_pure_phy = df_P6_pure_phy.reset_index(drop=True)
df_P6_combined_phy_chem = df_P6_combined_phy_chem.reset_index(drop=True)
df_P6_combined_bio_chem = df_P6_combined_bio_chem.reset_index(drop=True)
df_P6_combined_phy_bio = df_P6_combined_phy_bio.reset_index(drop=True)

print(df_P6_pure_bio.shape, df_P6_pure_chem.shape, df_P6_pure_phy.shape,
      df_P6_combined_phy_chem.shape, df_P6_combined_bio_chem.shape, df_P6_combined_phy_bio.shape)

# Upon merging, the row size will shrink as only those with the matching serial no. will retain
df_P6_combined_phy_chem.head()


# In[29]:


# Drop serial number columns
pure_bio.drop(columns=['Serial number'], inplace=True)
pure_chem.drop(columns=['Serial number'], inplace=True)
pure_phy.drop(columns=['Serial number'], inplace=True)
combined_phy_chem.drop(columns=['Serial number'], inplace=True)
combined_bio_chem.drop(columns=['Serial number'], inplace=True)
combined_phy_bio.drop(columns=['Serial number'], inplace=True)

df_P6_pure_bio.drop(columns=['Serial number'], inplace=True)
df_P6_pure_chem.drop(columns=['Serial number'], inplace=True)
df_P6_pure_phy.drop(columns=['Serial number'], inplace=True)
df_P6_combined_phy_chem.drop(columns=['Serial number'], inplace=True)
df_P6_combined_bio_chem.drop(columns=['Serial number'], inplace=True)
df_P6_combined_phy_bio.drop(columns=['Serial number'], inplace=True)


# # Data Visualization
# 
# ---
# 

# In[30]:


# Set up matplotlib figure with Two subplots
f, axes = plt.subplots(6, figsize=(24, 16))

axes[0].set_title('pure_bio')
sb.boxplot(data=pure_bio.iloc[:, :], orient="h", ax=axes[0])
axes[1].set_title('pure_chem')
sb.boxplot(data=pure_chem.iloc[:, :], orient="h", ax=axes[1])
axes[2].set_title('pure_phy')
sb.boxplot(data=pure_phy.iloc[:, :], orient="h", ax=axes[2])
axes[3].set_title('combined_phy_chem')
sb.boxplot(data=combined_phy_chem.iloc[:, :], orient="h", ax=axes[3])
axes[4].set_title('combined_bio_chem')
sb.boxplot(data=combined_bio_chem.iloc[:, :], orient="h", ax=axes[4])
axes[5].set_title('combined_phy_bio')
sb.boxplot(data=combined_phy_bio.iloc[:, :], orient="h", ax=axes[5])

# In[31]:


f, axes = plt.subplots(6, figsize=(24, 24))

axes[0].set_title('pure_bio')
sb.heatmap(data=pure_bio.corr(), vmin=-1, vmax=1,
           annot=True, fmt=".2f",  cmap="RdBu", ax=axes[0])
axes[1].set_title('pure_chem')
sb.heatmap(data=pure_chem.corr(), vmin=-1, vmax=1,
           annot=True, fmt=".2f",  cmap="RdBu", ax=axes[1])
axes[2].set_title('pure_phy')
sb.heatmap(data=pure_phy.corr(), vmin=-1, vmax=1,
           annot=True, fmt=".2f",  cmap="RdBu", ax=axes[2])
axes[3].set_title('combined_phy_chem')
sb.heatmap(data=combined_phy_chem.corr(), vmin=-1, vmax=1,
           annot=True, fmt=".2f",  cmap="RdBu", ax=axes[3])
axes[4].set_title('combined_bio_chem')
sb.heatmap(data=combined_bio_chem.corr(), vmin=-1, vmax=1,
           annot=True, fmt=".2f",  cmap="RdBu", ax=axes[4])
axes[5].set_title('combined_phy_bio')
sb.heatmap(data=combined_phy_bio.corr(), vmin=-1, vmax=1,
           annot=True, fmt=".2f",  cmap="RdBu", ax=axes[5])

# # Functions
# 
# ---
# 

# In[32]:


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


# In[33]:


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


# In[34]:


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


# In[35]:


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


# In[36]:


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


# In[37]:


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


# In[38]:


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


# In[39]:


def leeway(pred_res, allowance):
    lst = []
    for i in pred_res:
        actual = getresults_arr(i)
        plus = getresults_arr(i + allowance)
        minus = getresults_arr(i - allowance)
        lst.append([actual, plus, minus])
    return lst


# In[40]:


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


# Pure Bio data
cols = ['Sci_S1', 'Sci_S2', 'Bio_S4', 'Bio_S3', 'TScore']

# Calculate the quartiles
Q1 = combined_df[cols].quantile(0.25)
Q3 = combined_df[cols].quantile(0.75)

# Rule to identify outliers
IQR = Q3 - Q1

pure_bio = pure_bio[~((pure_bio[cols] < (
    Q1 - 1.5 * IQR)) | (pure_bio[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
df_P6_pure_bio = df_P6_pure_bio[~((df_P6_pure_bio[cols] < (
    Q1 - 1.5 * IQR)) | (df_P6_pure_bio[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
##########################################################################################
# Pure Chem data
cols = ['Sci_S1', 'Sci_S2', 'Chem_S4', 'Chem_S3', 'TScore']

# Calculate the quartiles
Q1 = combined_df[cols].quantile(0.25)
Q3 = combined_df[cols].quantile(0.75)

# Rule to identify outliers
IQR = Q3 - Q1

pure_chem = pure_chem[~((pure_chem[cols] < (
    Q1 - 1.5 * IQR)) | (pure_chem[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
df_P6_pure_chem = df_P6_pure_chem[~((df_P6_pure_chem[cols] < (
    Q1 - 1.5 * IQR)) | (df_P6_pure_chem[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
##########################################################################################
# Pure Phy data
cols = ['Sci_S1', 'Sci_S2', 'Phy_S4', 'Phy_S3', 'TScore']

# Calculate the quartiles
Q1 = combined_df[cols].quantile(0.25)
Q3 = combined_df[cols].quantile(0.75)

# Rule to identify outliers
IQR = Q3 - Q1

pure_phy = pure_phy[~((pure_phy[cols] < (
    Q1 - 1.5 * IQR)) | (pure_phy[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
df_P6_pure_phy = df_P6_pure_phy[~((pure_phy[cols] < (
    Q1 - 1.5 * IQR)) | (df_P6_pure_phy[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
##########################################################################################
# Combined Phy/Chem
cols = ['Sci_S1', 'Sci_S2',
        'Comb_Sci(Phy/Chem)_S4', 'Comb_Sci(Phy/Chem)_S3', 'TScore']

# Calculate the quartiles
Q1 = combined_df[cols].quantile(0.25)
Q3 = combined_df[cols].quantile(0.75)

# Rule to identify outliers
IQR = Q3 - Q1

combined_phy_chem = combined_phy_chem[~((combined_phy_chem[cols] < (
    Q1 - 1.5 * IQR)) | (combined_phy_chem[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
df_P6_combined_phy_chem = df_P6_combined_phy_chem[~((df_P6_combined_phy_chem[cols] < (
    Q1 - 1.5 * IQR)) | (df_P6_combined_phy_chem[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
##########################################################################################
# Combined Bio/Chem
cols = ['Sci_S1', 'Sci_S2',
        'Comb_Sci(Bio/Chem)_S4', 'Comb_Sci(Bio/Chem)_S3', 'TScore']

# Calculate the quartiles
Q1 = combined_df[cols].quantile(0.25)
Q3 = combined_df[cols].quantile(0.75)

# Rule to identify outliers
IQR = Q3 - Q1

combined_bio_chem = combined_bio_chem[~((combined_bio_chem[cols] < (
    Q1 - 1.5 * IQR)) | (combined_bio_chem[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
df_P6_combined_bio_chem = df_P6_combined_bio_chem[~((df_P6_combined_bio_chem[cols] < (
    Q1 - 1.5 * IQR)) | (df_P6_combined_bio_chem[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
##########################################################################################
# Combined Phy/Bio
cols = ['Sci_S1', 'Sci_S2',
        'Comb_Sci(Phy/Bio)_S4', 'Comb_Sci(Phy/Bio)_S3', 'TScore']

# Calculate the quartiles
Q1 = combined_df[cols].quantile(0.25)
Q3 = combined_df[cols].quantile(0.75)

# Rule to identify outliers
IQR = Q3 - Q1

combined_phy_bio = combined_phy_bio[~((combined_phy_bio[cols] < (
    Q1 - 1.5 * IQR)) | (combined_phy_bio[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
df_P6_combined_phy_bio = df_P6_combined_phy_bio[~((df_P6_combined_phy_bio[cols] < (
    Q1 - 1.5 * IQR)) | (df_P6_combined_phy_bio[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]


# # Linear Regression
# 

# In[41]:


# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# # Analysis of Variance (ANOVA) on linear models
# from statsmodels.stats.anova import anova_lm

# In[42]:


all_dfs= [pure_bio, pure_chem, pure_phy, combined_phy_chem, combined_bio_chem, combined_phy_bio]

# In[43]:


df_linreg = all_dfs[5].copy()

# To convert all these data into numerical value for decision tree
# Categorical data to scoring system
df_linreg['Gender'] = getGender(all_dfs[5])
df_linreg['Sec5_CardingLevel'] = getCarding(all_dfs[5])
df_linreg['Course'] = getcourse_tier(all_dfs[5])
df_linreg['ResidentialType'] = gethousing(all_dfs[5])
# Numerical data to scoring system
df_linreg['Sci'] = encode_psle(all_dfs[5], 'Sci')
# df_linreg = df_linreg[df_linreg['Comb_Sci(Phy/Bio)_S4'].notna()]
# df_linreg = df_linreg[df_linreg['Comb_Sci(Phy/Bio)_S3'].notna()]
# df_linreg = df_linreg[df_linreg['Sci_S2'].notna()]
# df_linreg = df_linreg[df_linreg['Sci_S1'].notna()]
# Columns that are applying KNN Imputer
cols = df_linreg.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
df_linreg = KNN_Imputer(df_linreg, cols)
df_linreg = pd.DataFrame(df_linreg, columns=cols)

# Validation
print(df_linreg.shape)


# In[44]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler()
# scaler.fit_transform()  # scales data
# scaler.inverse_transform()  # transforms scaled data back into original form
X = df_linreg[['TScore', 'Gender', 'Sec4_CardingLevel',
               'Course', 'Sci', 'ResidentialType', 'Comb_Sci(Phy/Bio)_S3', 'Sci_S2', 'Sci_S1',
                ]]
y = df_linreg[['Comb_Sci(Phy/Bio)_S4']]
# X = X.to_numpy()
# y = y.to_numpy()

X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=2)
y_test_final = scaler.inverse_transform(y_test).flatten()


# In[45]:


# Linear regression model
from sklearn.linear_model import LinearRegression
regmodel = LinearRegression().fit(X_train, y_train)


# In[46]:


# converting test results into grades
y_test_grade = []
for i in y_test_final:
    y_test_grade.append(getresults_arr(i))


# In[47]:


# Predictions from model
lr_preds = regmodel.predict(X_test)
# lr_preds = lr_preds.reshape(-1, 1)
lr_preds = scaler.inverse_transform(lr_preds).flatten().round()


# In[48]:


lr_preds_rg = leeway(lr_preds, 3)


# In[49]:


correct = 0
for i in range(len(lr_preds)):
    if y_test_grade[i] in lr_preds_rg[i]:
        correct += 1
print(correct / len(lr_preds))


# # Linear Regression for P6 Data
# 

# In[50]:


all_dfs_P6 = [df_P6_pure_bio, df_P6_pure_chem, df_P6_pure_phy, 
            df_P6_combined_phy_chem, df_P6_combined_bio_chem, df_P6_combined_phy_bio]

# In[51]:


df_test = all_dfs_P6[5].copy()
df_test['Gender'] = getGender(all_dfs_P6[5])
df_test['Sec4_CardingLevel'] = getCarding(all_dfs_P6[5])
df_test['Course'] = getcourse_tier(all_dfs_P6[5])
df_test['ResidentialType'] = gethousing(all_dfs_P6[5])

# Numerical data to scoring system
df_test['Sci'] = encode_psle(all_dfs_P6[5], 'Sci')
# df_test = df_test[df_test['Sci_S2'].notna()]
# df_test = df_test[df_test['Sci_S1'].notna()]

# Columns that are applying KNN Imputer
cols = df_test.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
df_test = KNN_Imputer(df_test, cols)
df_test = pd.DataFrame(df_test, columns=cols)

# Reset the df index
df_test.reset_index(inplace=True)
print(df_test.shape)


# In[52]:


X = df_test.copy()
print(X.shape)
X.drop(columns=['Comb_Sci(Phy/Bio)_S4'], inplace=True)
Y = y = df_test[['Comb_Sci(Phy/Bio)_S4']]
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=2)
y_test_final = scaler.inverse_transform(y_test).flatten()


# In[53]:


regmodel_p6 = LinearRegression().fit(X_train, y_train)


# In[54]:


# converting test results into grades
y_test_grade = []
for i in y_test_final:
    y_test_grade.append(getresults_arr(i))


# In[55]:


# Predictions from model
lr_predsp6 = regmodel_p6.predict(X_test)
# lr_predsp6 = lr_predsp6.reshape(-1, 1)
lr_predsp6 = scaler.inverse_transform(lr_predsp6).flatten().round()


# In[56]:


lr_predsp6_rg = leeway(lr_predsp6, 3)


# In[57]:


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

# In[58]:


# Temp to do the switch between Subjects
# var = pure_bio
# var = pure_chem
# var = pure_phy
# var = combined_phy_chem
# var = combined_bio_chem
var = combined_phy_bio

# Make a duplicate dataframe for manipulation
df_decisionTree_combined = var.copy()

# Columns that are applying KNN Imputer
cols = df_decisionTree_combined.columns.to_numpy()

# To convert all these data into numerical value for decision tree
# Categorical data to scoring system
df_decisionTree_combined['Gender'] = getGender(var)
df_decisionTree_combined['Sec4_CardingLevel'] = getCarding(var)
df_decisionTree_combined['Course'] = getcourse_tier(var)
df_decisionTree_combined['ResidentialType'] = gethousing(var)

# Numerical data to scoring system
df_decisionTree_combined['Sci'] = encode_psle(var, 'Sci')

# Banding System for Var
df_decisionTree_combined['Sci_S1'] = getresults(
    df_decisionTree_combined, 'Sci_S1')
df_decisionTree_combined['Sci_S2'] = getresults(
    df_decisionTree_combined, 'Sci_S2')
# df_decisionTree_combined['Bio_S3'] = getresults(
#     df_decisionTree_combined, 'Bio_S3')
# df_decisionTree_combined['Bio_S4'] = getresults(
#     df_decisionTree_combined, 'Bio_S4')
# df_decisionTree_combined['Chem_S3'] = getresults(
#     df_decisionTree_combined, 'Chem_S3')
# df_decisionTree_combined['Chem_S4'] = getresults(
#     df_decisionTree_combined, 'Chem_S4')
# df_decisionTree_combined['Phy_S3'] = getresults(
#     df_decisionTree_combined, 'Phy_S3')
# df_decisionTree_combined['Phy_S4'] = getresults(
#     df_decisionTree_combined, 'Phy_S4')
# df_decisionTree_combined['Comb_Sci(Phy/Chem)_S3'] = getresults(
#     df_decisionTree_combined, 'Comb_Sci(Phy/Chem)_S3')
# df_decisionTree_combined['Comb_Sci(Phy/Chem)_S4'] = getresults(
#     df_decisionTree_combined, 'Comb_Sci(Phy/Chem)_S4')
# df_decisionTree_combined['Comb_Sci(Bio/Chem)_S3'] = getresults(
#     df_decisionTree_combined, 'Comb_Sci(Bio/Chem)_S3')
# df_decisionTree_combined['Comb_Sci(Bio/Chem)_S4'] = getresults(
#     df_decisionTree_combined, 'Comb_Sci(Bio/Chem)_S4')
df_decisionTree_combined['Comb_Sci(Phy/Bio)_S3'] = getresults(
    df_decisionTree_combined, 'Comb_Sci(Phy/Bio)_S3')
df_decisionTree_combined['Comb_Sci(Phy/Bio)_S4'] = getresults(
    df_decisionTree_combined, 'Comb_Sci(Phy/Bio)_S4')

# # Fill up the NaN based on KNN Imputer
df_decisionTree_combined = KNN_Imputer(df_decisionTree_combined, cols)
df_decisionTree_combined = pd.DataFrame(df_decisionTree_combined, columns=cols)

# # Validation
print(df_decisionTree_combined.shape)
df_decisionTree_combined.head()


# In[59]:


# Temp to do the switch between Subjects
# var = df_P6_pure_bio
# var = df_P6_pure_chem
# var = df_P6_pure_phy
# var = df_P6_combined_phy_chem
# var = df_P6_combined_bio_chem
var = df_P6_combined_phy_bio

# Make a duplicate dataframe for manipulation
df_psych = var.copy()

# To convert all these data into numerical value for decision tree
# Categorical data to scoring system
df_psych['Gender'] = getGender(var)
df_psych['Sec4_CardingLevel'] = getCarding(var)
df_psych['Course'] = getcourse_tier(var)
df_psych['ResidentialType'] = gethousing(var)

# Columns that are applying KNN Imputer
cols = df_psych.columns.to_numpy()

# Numerical data to scoring system
df_psych['Sci'] = encode_psle(var, 'Sci')

# Banding System for Var
df_psych['Sci_S1'] = getresults(
    df_psych, 'Sci_S1')
df_psych['Sci_S2'] = getresults(
    df_psych, 'Sci_S2')
# df_psych['Bio_S3'] = getresults(
#     df_psych, 'Bio_S3')
# df_psych['Bio_S4'] = getresults(
#     df_psych, 'Bio_S4')
# df_psych['Chem_S3'] = getresults(
#     df_psych, 'Chem_S3')
# df_psych['Chem_S4'] = getresults(
#     df_psych, 'Chem_S4')
# df_psych['Phy_S3'] = getresults(
#     df_psych, 'Phy_S3')
# df_psych['Phy_S4'] = getresults(
#     df_psych, 'Phy_S4')
# df_psych['Comb_Sci(Phy/Chem)_S3'] = getresults(
#     df_psych, 'Comb_Sci(Phy/Chem)_S3')
# df_psych['Comb_Sci(Phy/Chem)_S4'] = getresults(
#     df_psych, 'Comb_Sci(Phy/Chem)_S4')
# df_psych['Comb_Sci(Bio/Chem)_S3'] = getresults(
#     df_psych, 'Comb_Sci(Bio/Chem)_S3')
# df_psych['Comb_Sci(Bio/Chem)_S4'] = getresults(
#     df_psych, 'Comb_Sci(Bio/Chem)_S4')
df_psych['Comb_Sci(Phy/Bio)_S3'] = getresults(
    df_psych, 'Comb_Sci(Phy/Bio)_S3')
df_psych['Comb_Sci(Phy/Bio)_S4'] = getresults(
    df_psych, 'Comb_Sci(Phy/Bio)_S4')

# Fill up the NaN based on KNN Imputer
df_psych = KNN_Imputer(df_psych, cols)
df_psych = pd.DataFrame(df_psych, columns=cols)

# # Validation
print(df_psych.shape)
df_psych.head()


# In[60]:


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
# | Variable                | Observation           | Description                                                                                            |
# | :---------------------- | :-------------------- | :----------------------------------------------------------------------------------------------------- |
# | Gender                  | Categorial : Nominal  | Changed to numeric (0,1) variable with no implication of quantitiative value or order                  |
# | Sec4_CardingLevel       | Categorial : Nominal  | Changed to boolean (False,True) variable with no implication of quantitiative value or order           |
# | Course                  | Categorial : Nominal  | Changed to numeic (1,2,3) to label variable with no implication of quantitiative value or order        |
# | EL                      | Categorial : Interval | Re-indexed Letter-Grade to Band to represent a numeric scale where the order and differences are known |
# | ResidentialType         | Categorial : Ordinal  | Labeled variable with a scale 1-to-8 for social-economical impact                                      |
# | TScore                  | Numerical : Ratio     | Used original range of 0 to 300                                                                        |
# | EL_S3                   | Numerical : Ratio     | Re-indexed to the last char of A1,A2,B3... banding system                                              |
# | EL_S2                   | Numerical : Ratio     | Re-indexed to the last char of A1,A2,B3... banding system                                              |
# | EL_S1                   | Numerical : Ratio     | Re-indexed to the last char of A1,A2,B3... banding system                                              |
# | _Respective predictors_ | Numerical : Ratio     | Re-indexed to the last char of A1,A2,B3... banding system                                              |
# | _Respective response_   | Numerical : Ratio     | Re-indexed for grade prediction, or as it is for score prediction                                      |
# 

# In[61]:


from sklearn.model_selection import train_test_split
# Temp to do the switch between subjects
# var = 'Bio_S4'
# var = 'Chem_S4'
# var = 'Phy_S4'
# var = 'Comb_Sci(Phy/Chem)_S4'
# var = 'Comb_Sci(Bio/Chem)_S4'
var = 'Comb_Sci(Phy/Bio)_S4'


# Extract Response and Predictors
X_combined = df_decisionTree_combined.drop([var], axis=1,
                                           inplace=False)  # Copy of df without predictor
y_combined = pd.DataFrame(df_decisionTree_combined[var])


# Split the Dataset into random Train and Test
X_combined_train, X_combined_test, y_combined_train, y_combined_test = train_test_split(
    X_combined, y_combined, test_size=0.3, random_state=2)

# Fit the model
dectree_combined.fit(X_combined_train, y_combined_train)

###############################################################################################
# Extract Response and Predictors
X_psych = df_psych.drop([var], axis=1, inplace=False)
y_psych = pd.DataFrame(df_psych[var])


# Split the Dataset into random Train and Test
X_psych_train, X_psych_test, y_psych_train, y_psych_test = train_test_split(
    X_psych, y_psych, test_size=0.3, random_state=2)

# Fit the model
dectree_psych.fit(X_psych_train, y_psych_train)


# In[62]:


# Plot the trained Decision Tree
from sklearn.tree import plot_tree

f = plt.figure(figsize=(24, 12))
# Left arrow: True
# Right arrow: False
# @para: class_names=['1', '2', '3', '4', '5', '6', '7', '8', '9']
plot_tree(dectree_combined, fontsize=10, filled=True, rounded=True,
          feature_names=X_combined_train.columns)


# In[63]:


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

# In[64]:


# Predict the Response corresponding to Predictors
y_combined_test_pred = dectree_combined.predict(X_combined_test)
y_psych_test_pred = dectree_psych.predict(X_psych_test)


# In[65]:


list_test = y_combined_test.to_numpy()
list_pred = y_combined_test_pred
correct = 0
for i in range(len(list_test)):
    if abs(list_test[i] - list_pred[i]) <= 0:  # Tweak val to change the accuracy
        correct += 1
print("Test Combined")
print("Accuracy : ", correct/len(list_test))


# In[66]:


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

# In[67]:


# Library to rescale variables into the range [0,1]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Supress warning for SettingWithCopyError
pd.options.mode.chained_assignment = None

# Temp to do the switch between subjects
var = pure_bio
# var = pure_chem
# var = pure_phy
# var = combined_phy_chem
# var = combined_bio_chem
# var = combined_phy_bio

# Dataset
df_DNN = var.copy().dropna()

# Columns that are applying KNN Imputer
cols = df_DNN.columns.tolist()

# Numerical data to scoring system
df_DNN['Sci'] = encode_psle(df_DNN, 'Sci')
# Convert from boolean to float
df_DNN['Sec4_CardingLevel'] = df_DNN['Sec4_CardingLevel'].astype(int)


# Banding System for Var
df_DNN['Sci_S1'] = getresults(
    df_DNN, 'Sci_S1')
df_DNN['Sci_S2'] = getresults(
    df_DNN, 'Sci_S2')
df_DNN['Bio_S3'] = getresults(
    df_DNN, 'Bio_S3')
df_DNN['Bio_S4'] = getresults(
    df_DNN, 'Bio_S4')

# Col will hold element that should be scaled
cols.remove('Course')
cols.remove('Sci')
cols.remove('Gender')
cols.remove('Sec4_CardingLevel')
cols.remove('ResidentialType')
print(cols)

# cols.to_numpy()
# Scale continuous variable in the dataset
for col in cols:
    df_DNN[col] = scaler.fit_transform(df_DNN[col].values.reshape(-1, 1))

# Rename columns because it violates the special char
# A scope name has to match the following pattern: ^[A-Za-z0-9_.\\/>-]*$
# df_DNN.rename(columns={
#               'Combined Humanities(SS/Geo)_S3': 'Combined_Humanities_S3'}, inplace=True)

# # Extract Response and Predictors
X = df_DNN.drop(columns=['Combined Humanities(SS/Geo)_S4'],
                inplace=False)  # Copy of df without predictor
y = pd.DataFrame(df_DNN['Combined Humanities(SS/Geo)_S4'])

# X needs to be in dataframe, y needs to be in series
y = y.squeeze()

# Split the Dataset into random Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[ ]:


# Library to rescale variables into the range [0,1]
from sklearn.preprocessing import MinMaxScaler

# Supress warning for SettingWithCopyError
pd.options.mode.chained_assignment = None

scaler = MinMaxScaler()

# Dataset
df_DNN = df_psych_original.copy().dropna()
# Extract Response and Predictors
X = df_DNN.drop(columns=['Bio_S4'], inplace=False)
# Convert Boolean to Integer (0,1)
X['Sec4_CardingLevel'] = X['Sec4_CardingLevel'].astype(int)
y = df_DNN.iloc[:, df_DNN.columns.get_loc("Bio_S4")]

# Scale continuous variable in the dataset
list_col = X.columns.tolist()
list_rmv = ['Gender', 'Sec4_CardingLevel', 'Course', 'EL', 'ResidentialType', 'Race', 'Academy']
for ele in list_rmv:
    list_col.remove(ele)
for ele in list_col:
    X[ele] = scaler.fit_transform(X[ele].values.reshape(-1, 1))
# Remove % symbol
X = X.rename(columns={'Percentage (%)': 'Percentage',
                      'Percentage (%).1': 'Percentage1',
                      'Percentage (%).2': 'Percentage2',
                      'Percentage (%).3': 'Percentage3',
                      'Percentage (%).4': 'Percentage4',
                      'Percentage (%).5': 'Percentage5',
                      'Q1 M': 'Q1_M',
                      'Q1.6': 'Q1_6',
                      'Q3.7': 'Q3_7',
                      'Percentage (%).5': 'Percentage5',
                      })
# for col in X.columns:
#     print(col)
X.drop(columns=['index'], inplace=True)
print(len(X.columns))
# y = scaler.fit_transform(y.values.reshape(-1, 1))

# Split the Dataset into random Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2)


# In[ ]:


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

# In[ ]:


# Columns
feature_columns = [
    tf.feature_column.indicator_column(Gender_column),
    tf.feature_column.indicator_column(Course_column),
    tf.feature_column.indicator_column(ResidentialType_column),
    tf.feature_column.numeric_column(
        key='Sec4_CardingLevel', dtype=tf.float64),
    tf.feature_column.numeric_column(key='EL', dtype=tf.float64),
    tf.feature_column.numeric_column(key='TScore', dtype=tf.float64),
    tf.feature_column.numeric_column(key='EL_S1', dtype=tf.float64),
    tf.feature_column.numeric_column(key='EL_S2', dtype=tf.float64),
    tf.feature_column.numeric_column(key='EL_S3', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Geo_S1', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Geo_S2', dtype=tf.float64),
    tf.feature_column.numeric_column(
        key='Combined_Humanities_S3', dtype=tf.float64),

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

# In[ ]:


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

# In[ ]:


model = tf.estimator.DNNRegressor(hidden_units=[32, 32, 32], feature_columns=feature_columns,
                                  optimizer=tf.optimizers.Adam(
                                      learning_rate=0.1),
                                  activation_fn=tf.nn.relu)


# In[ ]:


model.train(input_fn=input_func, steps=10000)


# Create a prediction input function and then use the .predict method to create a list or predictions on a test data.
# 

# In[ ]:


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

# In[ ]:


# Goodness of Fit of the DDN Model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print("Mean Squared Error: ", mean_squared_error(y_test, final_preds))
print("Mean Absolute Error: ", mean_absolute_error(y_test, final_preds))
print("R^2 Error: ", r2_score(y_test, final_preds))


# Comment: Mean absolute value shows that on average, the model makes a mistake of around 5~12 marks. This is considered as a poor result. To improve the model, consider inputing more data. Banding can also be an alternative when we're viewing the data.
# 

# Compare real values to predicted
# 

# In[ ]:


list_pred = []
for num in final_preds:
    list_pred.append(num[0])

d = {'Actual Result': y_test, 'Predicted by Model': list_pred}
df = pd.DataFrame(data=d)
df.round(2)[:10]  # Show 10 results


# In[ ]:


correct = 0
list_y = y_test.to_numpy()
for i in range(len(list_pred)):
    # allowance of 1 increases prediction accuracy
    if abs(list_pred[i] - list_y[i]) <= 1:
        correct += 1
print(correct / len(list_pred))


# In[ ]:




# # Multi Layer Perceptron
# 
# ---
# 
#     Feedforward artificial neural network (ANN)
# 

# In[ ]:


mlp_hum = all_dfs[5].copy()

# To convert all these data into numerical value for decision tree
# Categorical data to scoring system
mlp_hum['Gender'] = getGender(all_dfs[5])
mlp_hum['Sec5_CardingLevel'] = getCarding(all_dfs[5])
mlp_hum['Course'] = getcourse_tier(all_dfs[5])
mlp_hum['ResidentialType'] = gethousing(all_dfs[5])
# Numerical data to scoring system
mlp_hum['Sci'] = encode_psle(all_dfs[5], 'Sci')
# mlp_hum = mlp_hum[mlp_hum['Sci_S2'].notna()]
# mlp_hum = mlp_hum[mlp_hum['Sci_S1'].notna()]
# Columns that are applying KNN Imputer
cols = mlp_hum.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
mlp_hum = KNN_Imputer(mlp_hum, cols)
mlp_hum = pd.DataFrame(mlp_hum, columns=cols)

# Validation
print(mlp_hum.shape)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# scaler.fit_transform()  # scales data
# scaler.inverse_transform()  # transforms scaled data back into original form
X = mlp_hum[['TScore', 'Gender', 'Sec4_CardingLevel',
             'Course', 'Sci', 'ResidentialType', 'Comb_Sci(Phy/Bio)_S3', 'Sci_S2', 'Sci_S1',
            ]]
y = mlp_hum[['Comb_Sci(Phy/Bio)_S4']]
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


mlp_preds = leeway(preds, 3)


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
# # var = pure_bio
# # var = pure_chem
# # var = pure_phy
# # var = combined_phy_chem
# # var = combined_bio_chem
# var = combined_phy_bio

# # Dataset
# df_combined_BLSTM = var.copy()

# # Columns that are applying KNN Imputer
# cols = df_combined_BLSTM.columns.to_numpy()

# # Numerical data to scoring system
# df_combined_BLSTM['Sci'] = encode_psle(df_combined_BLSTM, 'Sci')

# # Convert all to numeric
# df_combined_BLSTM['Gender'] = getGender(df_combined_BLSTM)
# df_combined_BLSTM['Sec4_CardingLevel'] = getCarding(df_combined_BLSTM)
# df_combined_BLSTM['Course'] = getcourse_tier(df_combined_BLSTM)
# df_combined_BLSTM['ResidentialType'] = gethousing(df_combined_BLSTM)
# # Convert from boolean to float
# df_combined_BLSTM['Sec4_CardingLevel'] = df_combined_BLSTM['Sec4_CardingLevel'].astype(
#     int)

# # Banding System for Var
# df_combined_BLSTM['Sci_S1'] = getresults(
#     df_combined_BLSTM, 'Sci_S1')
# df_combined_BLSTM['Sci_S2'] = getresults(
#     df_combined_BLSTM, 'Sci_S2')
# # df_combined_BLSTM['Bio_S3'] = getresults(
# #     df_combined_BLSTM, 'Bio_S3')
# # df_combined_BLSTM['Bio_S4'] = getresults(
# #     df_combined_BLSTM, 'Bio_S4')
# # df_combined_BLSTM['Chem_S3'] = getresults(
# #     df_combined_BLSTM, 'Chem_S3')
# # df_combined_BLSTM['Chem_S4'] = getresults(
# #     df_combined_BLSTM, 'Chem_S4')
# # df_combined_BLSTM['Phy_S3'] = getresults(
# #     df_combined_BLSTM, 'Phy_S3')
# # df_combined_BLSTM['Phy_S4'] = getresults(
# #     df_combined_BLSTM, 'Phy_S4')
# # df_combined_BLSTM['Comb_Sci(Phy/Chem)_S3'] = getresults(
# #     df_combined_BLSTM, 'Comb_Sci(Phy/Chem)_S3')
# # df_combined_BLSTM['Comb_Sci(Phy/Chem)_S4'] = getresults(
# #     df_combined_BLSTM, 'Comb_Sci(Phy/Chem)_S4')
# # df_combined_BLSTM['Comb_Sci(Bio/Chem)_S3'] = getresults(
# #     df_combined_BLSTM, 'Comb_Sci(Bio/Chem)_S3')
# # df_combined_BLSTM['Comb_Sci(Bio/Chem)_S4'] = getresults(
#     # df_combined_BLSTM, 'Comb_Sci(Bio/Chem)_S4')
# df_combined_BLSTM['Comb_Sci(Phy/Bio)_S3'] = getresults(
#     df_combined_BLSTM, 'Comb_Sci(Phy/Bio)_S3')
# df_combined_BLSTM['Comb_Sci(Phy/Bio)_S4'] = getresults(
#     df_combined_BLSTM, 'Comb_Sci(Phy/Bio)_S4')

# # Fill up the NaN based on KNN Imputer
# df_combined_BLSTM = KNN_Imputer(df_combined_BLSTM, cols)
# df_combined_BLSTM = pd.DataFrame(df_combined_BLSTM, columns=cols)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

# Supress warning for SettingWithCopyError
pd.options.mode.chained_assignment = None

# Library to rescale variables into the range [0,1]
scaler = MinMaxScaler()

# Temp to do the switch between languages
# var = df_P6_pure_bio
# var = df_P6_pure_chem
# var = df_P6_pure_phy
# var = df_P6_combined_phy_chem
# var = df_P6_combined_bio_chem
var = df_P6_combined_phy_bio

# Dataset
df_psych_BLSTM = var.copy()
df_psych_BLSTM

# Columns that are applying KNN Imputer
cols = df_psych_BLSTM.columns.to_numpy()

# Numerical data to scoring system
df_psych_BLSTM['Sci'] = encode_psle(df_psych_BLSTM, 'Sci')

# Convert all to numeric
df_psych_BLSTM['Gender'] = getGender(df_psych_BLSTM)
df_psych_BLSTM['Sec4_CardingLevel'] = getCarding(df_psych_BLSTM)
df_psych_BLSTM['Course'] = getcourse_tier(df_psych_BLSTM)
df_psych_BLSTM['ResidentialType'] = gethousing(df_psych_BLSTM)
# Convert from boolean to float
df_psych_BLSTM['Sec4_CardingLevel'] = df_psych_BLSTM['Sec4_CardingLevel'].astype(
    int)

# Banding System for Var
df_psych_BLSTM['Sci_S1'] = getresults(
    df_psych_BLSTM, 'Sci_S1')
df_psych_BLSTM['Sci_S2'] = getresults(
    df_psych_BLSTM, 'Sci_S2')
# df_psych_BLSTM['Bio_S3'] = getresults(
#     df_psych_BLSTM, 'Bio_S3')
# df_psych_BLSTM['Bio_S4'] = getresults(
#     df_psych_BLSTM, 'Bio_S4')
# df_psych_BLSTM['Chem_S3'] = getresults(
#     df_psych_BLSTM, 'Chem_S3')
# df_psych_BLSTM['Chem_S4'] = getresults(
#     df_psych_BLSTM, 'Chem_S4')
# df_psych_BLSTM['Phy_S3'] = getresults(
#     df_psych_BLSTM, 'Phy_S3')
# df_psych_BLSTM['Phy_S4'] = getresults(
#     df_psych_BLSTM, 'Phy_S4')
# df_psych_BLSTM['Comb_Sci(Phy/Chem)_S3'] = getresults(
#     df_psych_BLSTM, 'Comb_Sci(Phy/Chem)_S3')
# df_psych_BLSTM['Comb_Sci(Phy/Chem)_S4'] = getresults(
#     df_psych_BLSTM, 'Comb_Sci(Phy/Chem)_S4')
# df_psych_BLSTM['Comb_Sci(Bio/Chem)_S3'] = getresults(
#     df_psych_BLSTM, 'Comb_Sci(Bio/Chem)_S3')
# df_psych_BLSTM['Comb_Sci(Bio/Chem)_S4'] = getresults(
#     df_psych_BLSTM, 'Comb_Sci(Bio/Chem)_S4')
df_psych_BLSTM['Comb_Sci(Phy/Bio)_S3'] = getresults(
    df_psych_BLSTM, 'Comb_Sci(Phy/Bio)_S3')
df_psych_BLSTM['Comb_Sci(Phy/Bio)_S4'] = getresults(
    df_psych_BLSTM, 'Comb_Sci(Phy/Bio)_S4')

# Fill up the NaN based on KNN Imputer
df_psych_BLSTM = KNN_Imputer(df_psych_BLSTM, cols)
df_psych_BLSTM = pd.DataFrame(df_psych_BLSTM, columns=cols)


# In[ ]:


# Temp to do the switch between subjects
# var = 'Bio_S4'
# var = 'Chem_S4'
# var = 'Phy_S4'
# var = 'Comb_Sci(Phy/Chem)_S4'
# var = 'Comb_Sci(Bio/Chem)_S4'
var = 'Comb_Sci(Phy/Bio)_S4'

# Extract Response and Predictors
# X = df_combined_BLSTM.drop(columns=[var])
# y = df_combined_BLSTM[var]

X = df_psych_BLSTM.drop(columns=[var])
y = df_psych_BLSTM[var]

# In[ ]:


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
              input_shape=(None, 50), return_sequences=True))  # Tweak the parameter to match the column size
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




# # XGBoost Regressor
# 
# - predicts numeric values
# 

# In[ ]:


xg_hum = all_dfs[5].copy()

# To convert all these data into numerical value for decision tree
# Categorical data to scoring system
xg_hum['Gender'] = getGender(all_dfs[5])
xg_hum['Sec5_CardingLevel'] = getCarding(all_dfs[5])
xg_hum['Course'] = getcourse_tier(all_dfs[5])
xg_hum['ResidentialType'] = gethousing(all_dfs[5])
# Numerical data to scoring system
xg_hum['Sci'] = encode_psle(all_dfs[5], 'Sci')
# xg_hum = xg_hum[xg_hum['Sci_S2'].notna()]
# xg_hum = xg_hum[xg_hum['Sci_S1'].notna()]
# Columns that are applying KNN Imputer
cols = xg_hum.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
xg_hum = KNN_Imputer(xg_hum, cols)
xg_hum = pd.DataFrame(xg_hum, columns=cols)

# Validation
print(xg_hum.shape)


# In[ ]:


from xgboost import XGBRegressor
scaler = MinMaxScaler()
# scaler.fit_transform()  # scales data
# scaler.inverse_transform()  # transforms scaled data back into original form
X = xg_hum[['TScore', 'Gender', 'Sec4_CardingLevel',
            'Course', 'Sci', 'ResidentialType', 'Comb_Sci(Phy/Bio)_S3', 'Sci_S2', 'Sci_S1',
            ]]
y = xg_hum[['Comb_Sci(Phy/Bio)_S4']]
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


df_test = all_dfs_P6[5].copy()
df_test['Gender'] = getGender(all_dfs_P6[5])
df_test['Sec4_CardingLevel'] = getCarding(all_dfs_P6[5])
df_test['Course'] = getcourse_tier(all_dfs_P6[5])
df_test['ResidentialType'] = gethousing(all_dfs_P6[5])

# Numerical data to scoring system
df_test['Sci'] = encode_psle(all_dfs_P6[5], 'Sci')
# df_test = df_test[df_test['Sci_S2'].notna()]
# df_test = df_test[df_test['Sci_S1'].notna()]

# Columns that are applying KNN Imputer
cols = df_test.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
df_test = KNN_Imputer(df_test, cols)
df_test = pd.DataFrame(df_test, columns=cols)

# Reset the df index
df_test.reset_index(inplace=True)
print(df_test.shape)


# In[ ]:


from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler()

X = df_test.copy()
print(X.shape)
X.drop(columns=['Comb_Sci(Phy/Bio)_S4'], inplace=True)
y = df_test[['Comb_Sci(Phy/Bio)_S4']]

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
print(correct / len(xg_p6preds))


# # MLP P6
# 

# In[ ]:


df_test = all_dfs_P6[5].copy()
df_test['Gender'] = getGender(all_dfs_P6[5])
df_test['Sec4_CardingLevel'] = getCarding(all_dfs_P6[5])
df_test['Course'] = getcourse_tier(all_dfs_P6[5])
df_test['ResidentialType'] = gethousing(all_dfs_P6[5])

# Numerical data to scoring system
df_test['Sci'] = encode_psle(all_dfs_P6[5], 'Sci')
# df_test = df_test[df_test['Sci_S2'].notna()]
# df_test = df_test[df_test['Sci_S1'].notna()]

# Columns that are applying KNN Imputer
cols = df_test.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
df_test = KNN_Imputer(df_test, cols)
df_test = pd.DataFrame(df_test, columns=cols)

# Reset the df index
df_test.reset_index(inplace=True)
print(df_test.shape)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler()

X = df_test.copy()
print(X.shape)
X.drop(columns=['Comb_Sci(Phy/Bio)_S4'], inplace=True)
# for col in X.columns:
#     print(col)
y = df_test[['Comb_Sci(Phy/Bio)_S4']]

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


mlp_p6preds_rg = leeway(mlp_p6preds, 3)


# In[ ]:


correct = 0
for i in range(len(mlp_p6preds)):
    if y_test_grade[i] in mlp_p6preds_rg[i]:
        correct += 1
        # print(y_test_grade[i], mlp_p6preds_rg[i])
print(correct / len(mlp_p6preds))
