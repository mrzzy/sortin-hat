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

# In[4]:


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

# In[5]:


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

# In[6]:


data_S4_2016.info()

# `describe()` provides you with statistical information about the data. This is another method.
# 

# In[7]:


data_S4_2016.describe()

# # Data Cleaning
# 
# ---
# 

# In[8]:


# Truncating dataframe to columns that we only need
df_S4_2016 = pd.DataFrame(data_S4_2016[['Serial number', 'Gender', 'Race', 'Academy', 'Sec4_CardingLevel', 'Course',
                          'TScore', 'EL', 'ResidentialType', 'English', 'English.1', 'English.2', 'English.3']])
df_S4_2017 = pd.DataFrame(data_S4_2017[['Serial number', 'Gender', 'Race', 'Academy', 'Sec4_CardingLevel', 'Course',
                          'TScore', 'EL', 'ResidentialType', 'English', 'English.1', 'English.2', 'English.3']])
df_S4_2018 = pd.DataFrame(data_S4_2018[['Serial number', 'Gender', 'Race', 'Academy', 'Sec4_CardingLevel', 'Course',
                          'TScore', 'EL', 'ResidentialType', 'English', 'English.1', 'English.2', 'English.3']])
df_S4_2019 = pd.DataFrame(data_S4_2019[['Serial number', 'Gender', 'Race', 'Academy', 'Sec4_CardingLevel', 'Course',
                          'TScore', 'EL', 'ResidentialType', 'English', 'English.1', 'English.2', 'English.3']])
df_S4_2020 = pd.DataFrame(data_S4_2020[['Serial number', 'Gender', 'Race', 'Academy', 'Sec4_CardingLevel', 'Course',
                          'TScore', 'EL', 'ResidentialType', 'English', 'English.1', 'English.2', 'English.3']])
df_S4_2021 = pd.DataFrame(data_S4_2021[['Serial number', 'Gender', 'Race', 'Academy', 'Sec4_CardingLevel', 'Course', 'TScore', 'EL', 'ResidentialType', 'English', 'English.1', 'English.2', 'English.3']])

# Renaming to avoid confusion
df_S4_2016 = df_S4_2016.rename(columns={
                               'English': 'English_S4', 'English.1': 'English_S3', 'English.2': 'English_S2', 'English.3': 'English_S1'})
df_S4_2017 = df_S4_2017.rename(columns={
                               'English': 'English_S4', 'English.1': 'English_S3', 'English.2': 'English_S2', 'English.3': 'English_S1'})
df_S4_2018 = df_S4_2018.rename(columns={
                               'English': 'English_S4', 'English.1': 'English_S3', 'English.2': 'English_S2', 'English.3': 'English_S1'})
df_S4_2019 = df_S4_2019.rename(columns={
                               'English': 'English_S4', 'English.1': 'English_S3', 'English.2': 'English_S2', 'English.3': 'English_S1'})
df_S4_2020 = df_S4_2020.rename(columns={
                               'English': 'English_S4', 'English.1': 'English_S3', 'English.2': 'English_S2', 'English.3': 'English_S1'})
df_S4_2021 = df_S4_2021.rename(columns={
                               'English': 'English_S4', 'English.1': 'English_S3', 'English.2': 'English_S2', 'English.3': 'English_S1'})


# In[9]:


# cleaning for 2021 data
# replace all dashes with 0
col_2021 = ['English_S3', 'English_S2', 'English_S1']
df_S4_2021[col_2021] = df_S4_2021[col_2021].replace({'-': 0})
# turn columns into numeric values
df_S4_2021['English_S3'] = pd.to_numeric(df_S4_2021['English_S3'])
df_S4_2021['English_S2'] = pd.to_numeric(df_S4_2021['English_S2'])
df_S4_2021['English_S1'] = pd.to_numeric(df_S4_2021['English_S1'])


# In[14]:


# Merging all dfs together (Row order is preserved)
dfs = [df_S4_2016, df_S4_2017, df_S4_2018, df_S4_2019, df_S4_2020, df_S4_2021]
combined_df = pd.concat(dfs)
combined_df = combined_df.reset_index(drop=True)

# Standardize nan
cols = ['English_S4', 'English_S3', 'English_S2', 'English_S1']
combined_df[cols] = combined_df[cols].replace({0: np.nan, 0: np.nan})

# Setting boolean for carded athlete
col = 'Sec4_CardingLevel'
combined_df[col] = combined_df[col].replace(np.nan, False)
combined_df[col] = combined_df[col].replace(
    ['L3', 'Y', 'L4P', 'L4', 'YT', 'TL3', 'E3', 'B4', 'ET3', 'Y+'], True)
print(combined_df[col].value_counts())

# Seperating df based on stream
combined_df_express = combined_df[combined_df['Course'] == 'Express']
combined_df_normalAcademic = combined_df[combined_df['Course']
                                         == 'Normal Academic']

# Notice the min is not zero, but actually there's nan inside the dataset. Hence, the count is different
combined_df.describe()


# ## DataFrame:
# 
#     Merging P6 Screening Result with Secondary School Result
#     P6 2016 + Secondary 2020
#     P6 2017 + Secondary 2021
# 

# In[15]:


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


# In[16]:


# Convert the column to type float so that we can merge
data_P6_2016['Serial number'] = data_P6_2016['Serial number'].astype(int)
data_P6_2017['Serial number'] = data_P6_2017['Serial number'].astype(int)

# Renaming the data
data_S4_2020 = data_S4_2020.rename(columns={
    'English': 'English_S4', 'English.1': 'English_S3', 'English.2': 'English_S2', 'English.3': 'English_S1'})
data_S4_2021 = data_S4_2021.rename(columns={
    'English': 'English_S4', 'English.1': 'English_S3', 'English.2': 'English_S2', 'English.3': 'English_S1'})

# Merging of data (Paring of S/No. is unique)
df_P6_with_S4_2020 = data_P6_2016.merge(
    combined_df, how='inner', on=['Serial number'])
df_P6_with_S4_2021 = data_P6_2017.merge(
    combined_df, how='inner', on=['Serial number'])
print(df_P6_with_S4_2021.columns)


# In[17]:


# Check the dataset is clean
for column in combined_df:
    print(combined_df[column].value_counts())
    print("=======================================================")

# Seperating df based on stream
combined_df_express = combined_df[combined_df['Course'] == 'Express']
combined_df_normalAcademic = combined_df[combined_df['Course']
                                         == 'Normal Academic']
# Removal of serial number from combined df because df_P6_with_S4 has gotten what it needs
combined_df.drop(columns=['Serial number'], inplace=True)
combined_df_express.drop(columns=['Serial number'], inplace=True)
combined_df_normalAcademic.drop(columns=['Serial number'], inplace=True)


# # Data Visualization
# 
# ---
# 

# In[18]:


# Set up matplotlib figure with Two subplots
f, axes = plt.subplots(2, figsize=(24, 12))
xticks = [0, 50, 100, 150, 200, 250, 300]

axes[0].set_title('combined_df_express')
g1 = sb.boxplot(data=combined_df_express.iloc[:, 5:], orient="h", ax=axes[0])
axes[1].set_title('combined_df_normalAcademic')
g2 = sb.boxplot(
    data=combined_df_normalAcademic.iloc[:, 5:], orient="h", ax=axes[1])


# In[19]:


# Calculate the correlation between the numeric variables
combined_df_express.corr()


# In[20]:


# Calculate the correlation between the numeric variables
combined_df_normalAcademic.corr()


# In[21]:


f, axes = plt.subplots(3, figsize=(24, 12))
axes[0].set_title('combined_df')
sb.heatmap(combined_df.corr(), vmin=-1, vmax=1,
           annot=True, fmt=".2f",  cmap="RdBu", ax=axes[0])
axes[1].set_title('combined_df_express')
sb.heatmap(combined_df_express.corr(), vmin=-1, vmax=1,
           annot=True, fmt=".2f",  cmap="RdBu", ax=axes[1])
axes[2].set_title('combined_df_normalAcademic')
sb.heatmap(combined_df_normalAcademic.corr(), vmin=-1, vmax=1,
           annot=True, fmt=".2f",  cmap="RdBu", ax=axes[2])


# # Functions
# 
# ---
# 

# In[64]:


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


# In[65]:


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


# In[66]:


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


# In[67]:


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


# In[68]:


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


# In[69]:


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


# In[70]:


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


# In[71]:


def leeway(pred_res, allowance):
    lst = []
    for i in pred_res:
        actual = getresults_arr(i)
        plus = getresults_arr(i + allowance)
        minus = getresults_arr(i - allowance)
        lst.append([actual, plus, minus])
    return lst


# In[ ]:


from sklearn.impute import KNNImputer
# Comment: Simple Imputer will yield the same result


def KNN_Imputer(df, cols):
    imputer = KNNImputer(n_neighbors=3)
    for col in cols:
        df[col] = imputer.fit_transform(df[[col]])
        df[col] = np.around(df[col], 0)
    return df


# In[ ]:


# Merge the dataframe
df_psych_original = pd.concat([df_P6_with_S4_2020, df_P6_with_S4_2021])

# Specify the columns that we're working with
df_psych_original = df_psych_original[['Percentage (%)', 'Percentage (%).1', 'Percentage (%).2', 'Percentage (%).3', 'Percentage (%).4', 'Percentage (%).5', 'TScore', 'Gender', 'Sec4_CardingLevel', 'Race', 'Academy',
                                       'Course', 'EL', 'ResidentialType', 'English_S4', 'English_S3', 'English_S2', 'English_S1', 'Q1 M', 'Q1F', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8a', 'Q8b', 'Q8c', 'Q1.6', 'Q2a', 'Q2b', 'Q2c', 'Q2d', 'Q2e', 'Q2f',
                                       'Q2g', 'Q2h', 'Q2i', 'Q2j', 'Q2k', 'Q3.7', 'Q4a', 'Q4b', 'Q4c', 'Q4d',
                                       'Q4e', 'Q4f', 'Q4g', 'Q4h', 'Q4i', 'Q4j', 'Q4k']]

# Removal of serial number from combined df because df_P6_with_S4 has gotten what it needs
combined_df.drop(columns=['Serial number'], inplace=True)
combined_df_express.drop(columns=['Serial number'], inplace=True)
combined_df_normalAcademic.drop(columns=['Serial number'], inplace=True)

# # Remove Outlier

# In[ ]:


cols = ['TScore', 'English_S4', 'English_S3', 'English_S2', 'English_S1'] # one or more

# Calculate the quartiles
Q1 = combined_df[cols].quantile(0.25)
Q3 = combined_df[cols].quantile(0.75)
# Rule to identify outliers
IQR = Q3 - Q1

combined_df = combined_df[~((combined_df[cols] < (Q1 - 1.5 * IQR)) |(combined_df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
df_psych_original = df_psych_original[~((df_psych_original[cols] < (Q1 - 1.5 * IQR)) |(df_psych_original[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
combined_df


# # Linear Regression
# 

# In[ ]:


df_linreg = combined_df.copy()

# To convert all these data into numerical value for decision tree
# Categorical data to scoring system
df_linreg['Gender'] = getGender(combined_df)
df_linreg['Sec4_CardingLevel'] = getCarding(combined_df)
df_linreg['Course'] = getcourse_tier(combined_df)
df_linreg['ResidentialType'] = gethousing(combined_df)
# Numerical data to scoring system
df_linreg['EL'] = encode_psle(combined_df, 'EL')
# df_linreg = df_linreg[df_linreg['English_S4'].notna()]
# df_linreg = df_linreg[df_linreg['English_S3'].notna()]
# df_linreg = df_linreg[df_linreg['English_S2'].notna()]
# df_linreg = df_linreg[df_linreg['English_S1'].notna()]
# Columns that are applying KNN Imputer
cols = df_linreg.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
df_linreg = KNN_Imputer(df_linreg, cols)
df_linreg = pd.DataFrame(df_linreg, columns=cols)

# Validation
print(df_linreg.shape)

# In[ ]:


# to ignore if want to include express and NA
df_linreg = df_linreg[df_linreg['Course'] == 2]
print(df_linreg.shape)

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# scaler.fit_transform()  # scales data
# scaler.inverse_transform()  # transforms scaled data back into original form
X = df_linreg[['TScore', 'Gender', 'Sec4_CardingLevel',
               'Course', 'EL', 'ResidentialType', 'English_S3', 'English_S2', 'English_S1']]
y = df_linreg[['English_S4']]
# X = X.to_numpy()
# y = y.to_numpy()

X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=2)
y_test_final = scaler.inverse_transform(y_test).flatten()

# In[ ]:


# Linear regression model
from sklearn.linear_model import LinearRegression
regmodel = LinearRegression().fit(X_train, y_train)

# In[ ]:


# converting test results into grades
y_test_grade = []
for i in y_test_final:
    y_test_grade.append(getresults_arr(i))

# In[ ]:


# Predictions from model
lr_preds = regmodel.predict(X_test)
# lr_preds = lr_preds.reshape(-1, 1)
lr_preds = scaler.inverse_transform(lr_preds).flatten().round()

# In[ ]:


lr_preds_rg = leeway(lr_preds, 3)

# In[ ]:


correct = 0
for i in range(len(lr_preds)):
    if y_test_grade[i] in lr_preds_rg[i]:
        correct += 1
print(correct / len(lr_preds))

# # Linear Regression for P6 Data
# 

# In[ ]:


# Make a duplicate dataframe for manipulation
df_test = df_psych_original.copy()

# To convert all these data into numerical value for decision tree
# Categorical data to scoring system
df_test['Gender'] = getGender(df_psych_original)
df_test['Sec4_CardingLevel'] = getCarding(df_psych_original)
df_test['Course'] = getcourse_tier(df_psych_original)
df_test['ResidentialType'] = gethousing(df_psych_original)

# Numerical data to scoring system
df_test['EL'] = encode_psle(df_psych_original, 'EL')
# df_test = df_test[df_test['English_S4'].notna()]
# df_test = df_test[df_test['English_S3'].notna()]
# df_test = df_test[df_test['English_S2'].notna()]
# df_test = df_test[df_test['English_S1'].notna()]

# Columns that are applying KNN Imputer
cols = df_test.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
df_test = KNN_Imputer(df_test, cols)
df_test = pd.DataFrame(df_test, columns=cols)

# Reset the df index
df_test.reset_index(inplace=True)

# Restoring the 2 un-affected columns
df_test['Race'] = df_psych_original['Race'].copy()
df_test['Academy'] = df_psych_original['Academy'].copy()

# Validation
print(df_test.shape)
df_test.head(5)
df_test1 = pd.get_dummies(df_test, columns=[
    'Race', 'Academy'], drop_first=True)

# In[ ]:


X = df_test1.copy()
print(X.shape)
X.drop(columns=['English_S4'], inplace=True)
Y = y = df_test1[['English_S4']]
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=2)
y_test_final = scaler.inverse_transform(y_test).flatten()

# In[ ]:


regmodel_p6 = LinearRegression().fit(X_train, y_train)

# In[ ]:


# converting test results into grades
y_test_grade = []
for i in y_test_final:
    y_test_grade.append(getresults_arr(i))

# In[ ]:


# Predictions from model
lr_predsp6 = regmodel_p6.predict(X_test)
# lr_predsp6 = lr_predsp6.reshape(-1, 1)
lr_predsp6 = scaler.inverse_transform(lr_predsp6).flatten().round()

# In[ ]:


lr_predsp6_rg = leeway(lr_predsp6, 5)

# In[ ]:


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

# In[ ]:


# Make a duplicate dataframe for manipulation
df_decisionTree_combined = combined_df.copy()

# To convert all these data into numerical value for decision tree
# Categorical data to scoring system
df_decisionTree_combined['Gender'] = getGender(combined_df)
df_decisionTree_combined['Sec4_CardingLevel'] = getCarding(combined_df)
df_decisionTree_combined['Course'] = getcourse_tier(combined_df)
df_decisionTree_combined['ResidentialType'] = gethousing(combined_df)

# Numerical data to scoring system
df_decisionTree_combined['EL'] = encode_psle(combined_df, 'EL')
df_decisionTree_combined['English_S4'] = getresults(combined_df, 'English_S4')
df_decisionTree_combined['English_S3'] = getresults(combined_df, 'English_S3')
df_decisionTree_combined['English_S2'] = getresults(combined_df, 'English_S2')
df_decisionTree_combined['English_S1'] = getresults(combined_df, 'English_S1')

# Columns that are applying KNN Imputer
cols = df_decisionTree_combined.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
df_decisionTree_combined = KNN_Imputer(df_decisionTree_combined, cols)
df_decisionTree_combined = pd.DataFrame(df_decisionTree_combined, columns=cols)

# Restoring the 2 un-affected columns
df_decisionTree_combined['Race'] = combined_df['Race'].copy()
df_decisionTree_combined['Academy'] = combined_df['Academy'].copy()

# Validation
print(df_decisionTree_combined.shape)
df_decisionTree_combined.head(5)

# Backup
df_decisionTree_backup = df_decisionTree_combined.copy()

# In[ ]:


# Make a duplicate dataframe for manipulation
df_psych = df_psych_original.copy()

# To convert all these data into numerical value for decision tree
# Categorical data to scoring system
df_psych['Gender'] = getGender(df_psych_original)
df_psych['Sec4_CardingLevel'] = getCarding(df_psych_original)
df_psych['Course'] = getcourse_tier(df_psych_original)
df_psych['ResidentialType'] = gethousing(df_psych_original)

# Numerical data to scoring system
df_psych['EL'] = encode_psle(df_psych_original, 'EL')
df_psych['English_S4'] = getresults(df_psych_original, 'English_S4')
df_psych['English_S3'] = getresults(df_psych_original, 'English_S3')
df_psych['English_S2'] = getresults(df_psych_original, 'English_S2')
df_psych['English_S1'] = getresults(df_psych_original, 'English_S1')

# Columns that are applying KNN Imputer
cols = df_psych.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
df_psych = KNN_Imputer(df_psych, cols)
df_psych = pd.DataFrame(df_psych, columns=cols)

# Reset the df index
df_psych_original.reset_index(inplace=True)

# Restoring the 2 un-affected columns
df_psych['Race'] = df_psych_original['Race'].copy()
df_psych['Academy'] = df_psych_original['Academy'].copy()

# Validation
print(df_psych.shape)
df_psych.head(5)


# In[ ]:


# Import Decision Tree Classifier model from Scikit-Learn
from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree Classifier object (Do experiment with the max_depth)
dectree_combined = DecisionTreeClassifier(max_depth=2)
dectree_psych = DecisionTreeClassifier(max_depth=2)

# Apply One Hot Encoder (Categorical)
df_decisionTree_combined = pd.get_dummies(df_decisionTree_combined, columns=[
    'Race', 'Academy'], drop_first=True)
# df_decisionTree_psych = pd.get_dummies(df_psych, columns=[
df_decisionTree_psych = pd.get_dummies(df_psych, columns=[
    'Race', 'Academy'], drop_first=True)
# Note the column size will increase as per One Hot Encoder's work


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
# | EL                | Categorial : Interval | Re-indexed Letter-Grade to Band to represent a numeric scale where the order and differences are known |
# | ResidentialType   | Categorial : Ordinal  | Labeled variable with a scale 1-to-8 for social-economical impact                                      |
# | TScore            | Numerical : Ratio     | Used original range of 0 to 300                                                                        |
# | English_S4        | Numerical : Ratio     | Re-indexed to the last char of A1,A2,B3... banding system                                              |
# | English_S3        | Numerical : Ratio     | Re-indexed to the last char of A1,A2,B3... banding system                                              |
# | English_S2        | Numerical : Ratio     | Re-indexed to the last char of A1,A2,B3... banding system                                              |
# | English_S1        | Numerical : Ratio     | Re-indexed to the last char of A1,A2,B3... banding system                                              |
# 

# In[ ]:


from sklearn.model_selection import train_test_split

# Extract Response and Predictors
X_combined = df_decisionTree_combined.drop(['English_S4'], axis=1,
                                  inplace=False)  # Copy of df without predictor
y_combined = pd.DataFrame(df_decisionTree_combined['English_S4'])
print(y_combined)

# Split the Dataset into random Train and Test
X_combined_train, X_combined_test, y_combined_train, y_combined_test = train_test_split(
    X_combined, y_combined, test_size=0.3, random_state=2)

# Fit the model
dectree_combined.fit(X_combined_train, y_combined_train)

###############################################################################################
# Extract Response and Predictors
X_psych = df_decisionTree_psych.drop(['English_S4'], axis=1,
                                  inplace=False)  # Copy of df without predictor
y_psych = pd.DataFrame(df_decisionTree_psych['English_S4'])

# Split the Dataset into random Train and Test
X_psych_train, X_psych_test, y_psych_train, y_psych_test = train_test_split(
    X_psych, y_psych, test_size=0.2, random_state=2)

# Fit the model
dectree_psych.fit(X_psych_train, y_psych_train)


# In[ ]:


# Plot the trained Decision Tree
from sklearn.tree import plot_tree

f = plt.figure(figsize=(24, 12))
# Left arrow: True
# Right arrow: False
# @para: class_names=['1', '2', '3', '4', '5', '6', '7', '8', '9']
plot_tree(dectree_combined, fontsize=10, filled=True, rounded=True,
          feature_names=X_combined_train.columns)

# In[ ]:


# Plot the trained Decision Tree
from sklearn.tree import plot_tree

f = plt.figure(figsize=(24, 12))
# Left arrow: True
# Right arrow: False
# @para: class_names=['1', '2', '3', '4', '5', '6', '7', '8', '9']
plot_tree(dectree_combined, fontsize=10, filled=True, rounded=True,
          feature_names=X_combined_train.columns)


# #### Prediction on Train Data and Goodness of Fit
# 
# Check how good the predictions are on the Train Set.  
# Metrics : Classification Accuracy and Confusion Matrix.
# 

# In[ ]:


# Import the required metric from sklearn
from sklearn.metrics import confusion_matrix

# Predict the Response corresponding to Predictors
y_combined_train_pred = dectree_combined.predict(X_combined_train)

# Plot the two-way Confusion Matrix
fig = plt.subplots(figsize=(12,12))   
sb.heatmap(confusion_matrix(y_combined_train, y_combined_train_pred),
           annot=True, fmt=".0f", annot_kws={"size": 12})



# In[ ]:


# Predict the Response corresponding to Predictors
y_psych_train_pred = dectree_psych.predict(X_psych_train)

# Plot the two-way Confusion Matrix
fig = plt.subplots(figsize=(12,12))
sb.heatmap(confusion_matrix(y_psych_train, y_psych_train_pred),
           annot=True, fmt=".0f", annot_kws={"size": 12})

# In[ ]:


# # Print the Classification Accuracy
# print("Train Data (Combined)")
# # Shouldn't use the library accuracy model as we're not doing binary classification => dectree_combined.score(x_train, y_train)
# print()

# # Print the Accuracy Measures from the Confusion Matrix
# cmTrain = confusion_matrix(y_combined_train, y_combined_train_pred)

# FP = cmTrain.sum(axis=0) - np.diag(cmTrain)
# FN = cmTrain.sum(axis=1) - np.diag(cmTrain)
# TP = np.diag(cmTrain)
# TN = cmTrain.sum() - (FP + FN + TP)

# # Sensitivity, hit rate, recall, or true positive rate
# TPR = TP/(TP+FN)
# # Specificity or true negative rate
# TNR = TN/(TN+FP)
# # Fall out or false positive rate
# FPR = FP/(FP+TN)
# # False negative rate
# FNR = FN/(TP+FN)
# # Overall accuracy
# # ACC = (TP+TN)/(TP+FP+FN+TN)
# ACC = (TP)/(TP+FP)

# print("Accuracy:\t\t\t", np.round(ACC, 2))
# print()
# print("True Positive Rate :\t\t", np.round(TPR, 2))
# print("True Negative Rate :\t\t", np.round(TNR, 2))
# print()
# print("False Positive Rate :\t\t", np.round(FPR, 2))
# print("False Negative Rate :\t\t", np.round(FNR, 2))


# In[ ]:


list_train = y_combined_train.to_numpy()
list_pred = y_combined_train_pred
correct = 0
for i in range(len(list_train)):
    if abs(list_train[i] - list_pred[i]) <=3: # Tweak val to change the accuracy
        correct += 1
print("Train Combined")
print("Accuracy : ",correct/len(list_train))

# In[ ]:


# # Print the Classification Accuracy
# print("Train Data (Psych)")
# # Shouldn't use the library accuracy model as we're not doing binary classification => dectree_combined.score(x_train, y_train)
# print()

# # Print the Accuracy Measures from the Confusion Matrix
# cmTrain = confusion_matrix(y_psych_train, y_psych_train_pred)

# FP = cmTrain.sum(axis=0) - np.diag(cmTrain)
# FN = cmTrain.sum(axis=1) - np.diag(cmTrain)
# TP = np.diag(cmTrain)
# TN = cmTrain.sum() - (FP + FN + TP)

# # Sensitivity, hit rate, recall, or true positive rate
# TPR = TP/(TP+FN)
# # Specificity or true negative rate
# TNR = TN/(TN+FP)
# # Fall out or false positive rate
# FPR = FP/(FP+TN)
# # False negative rate
# FNR = FN/(TP+FN)
# # Overall accuracy
# # ACC = (TP+TN)/(TP+FP+FN+TN)
# ACC = (TP)/(TP+FP)

# print("Accuracy:\t\t\t", np.round(ACC, 2))
# print()
# print("True Positive Rate :\t\t", np.round(TPR, 2))
# print("True Negative Rate :\t\t", np.round(TNR, 2))
# print()
# print("False Positive Rate :\t\t", np.round(FPR, 2))
# print("False Negative Rate :\t\t", np.round(FNR, 2))


# In[ ]:


list_train = y_psych_train.to_numpy()
list_pred = y_psych_train_pred
correct = 0
for i in range(len(list_train)):
    if abs(list_train[i] - list_pred[i]) <=3: # Tweak val to change the accuracy
        correct += 1
print("Train psych")
print("Accuracy : ",correct/len(list_train))

# #### Prediction on Test Data and Goodness of Fit
# 
# Check how good the predictions are on the Test Set.  
# Metrics : Classification Accuracy and Confusion Matrix.
# 

# In[ ]:


# Predict the Response corresponding to Predictors
y_combined_test_pred = dectree_combined.predict(X_combined_test)

# Plot the two-way Confusion Matrix
fig = plt.subplots(figsize=(12,12))
sb.heatmap(confusion_matrix(y_combined_test, y_combined_test_pred),
           annot=True, fmt=".0f", annot_kws={"size": 12})


# In[ ]:


# Predict the Response corresponding to Predictors
y_psych_test_pred = dectree_psych.predict(X_psych_test)

# Plot the two-way Confusion Matrix
fig = plt.subplots(figsize=(12,12))
sb.heatmap(confusion_matrix(y_psych_test, y_psych_test_pred),
           annot=True, fmt=".0f", annot_kws={"size": 12})


# In[ ]:


# # Print the Classification Accuracy
# print("Test Data (Combined)")
# print()

# # Print the Accuracy Measures from the Confusion Matrix
# cmTest = confusion_matrix(y_combined_test, y_combined_test_pred)
# FP = cmTest.sum(axis=0) - np.diag(cmTest)
# FN = cmTest.sum(axis=1) - np.diag(cmTest)
# TP = np.diag(cmTest)
# TN = cmTest.sum() - (FP + FN + TP)

# # Sensitivity, hit rate, recall, or true positive rate
# TPR = TP/(TP+FN)
# # Specificity or true negative rate
# TNR = TN/(TN+FP)
# # Fall out or false positive rate
# FPR = FP/(FP+TN)
# # False negative rate
# FNR = FN/(TP+FN)
# # Overall accuracy
# # ACC = (TP+TN)/(TP+FP+FN+TN)
# ACC = TP/(TP+FP)

# print("Accuracy:\t\t\t", np.round(ACC, 2))
# print()
# print("True Positive Rate :\t\t", np.round(TPR, 2))
# print("True Negative Rate :\t\t", np.round(TNR, 2))
# print()
# print("False Positive Rate :\t\t", np.round(FPR, 2))
# print("False Negative Rate :\t\t", np.round(FNR, 2))


# In[ ]:


list_test = y_combined_test.to_numpy()
list_pred = y_combined_test_pred
correct = 0
for i in range(len(list_test)):
    if abs(list_test[i] - list_pred[i]) <=1: # Tweak val to change the accuracy
        correct += 1
print("Test Combined")
print("Accuracy : ",correct/len(list_test))

# In[ ]:


# # Print the Classification Accuracy
# print("Test Data (Psych)")
# print()

# # Print the Accuracy Measures from the Confusion Matrix
# cmTest = confusion_matrix(y_psych_test, y_psych_test_pred)
# FP = cmTest.sum(axis=0) - np.diag(cmTest)
# FN = cmTest.sum(axis=1) - np.diag(cmTest)
# TP = np.diag(cmTest)
# TN = cmTest.sum() - (FP + FN + TP)

# # Sensitivity, hit rate, recall, or true positive rate
# TPR = TP/(TP+FN)
# # Specificity or true negative rate
# TNR = TN/(TN+FP)
# # Fall out or false positive rate
# FPR = FP/(FP+TN)
# # False negative rate
# FNR = FN/(TP+FN)
# # Overall accuracy
# # ACC = (TP+TN)/(TP+FP+FN+TN)
# ACC = (TP)/(TP+FP)

# print("Accuracy:\t\t\t", np.round(ACC, 2))
# print()
# print("True Positive Rate :\t\t", np.round(TPR, 2))
# print("True Negative Rate :\t\t", np.round(TNR, 2))
# print()
# print("False Positive Rate :\t\t", np.round(FPR, 2))
# print("False Negative Rate :\t\t", np.round(FNR, 2))


# In[ ]:


list_test = y_psych_test.to_numpy()
list_pred = y_psych_test_pred
correct = 0
for i in range(len(list_test)):
    if abs(list_test[i] - list_pred[i]) <=1: # Tweak val to change the accuracy
        correct += 1
print("Test Psych")
print("Accuracy : ",correct/len(list_test))

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

# In[ ]:


# # Library to rescale variables into the range [0,1]
# from sklearn.preprocessing import MinMaxScaler

# # Supress warning for SettingWithCopyError
# pd.options.mode.chained_assignment = None

# scaler = MinMaxScaler()

# # Dataset
# df_DNN = combined_df.copy().dropna()

# # Extract Response and Predictors
# X = df_DNN[['Gender', 'Race', 'Academy', 'Sec4_CardingLevel', 'Course',
#             'TScore', 'EL', 'ResidentialType', 'English_S3', 'English_S2', 'English_S1']]
# # Convert Boolean to Integer (0,1)
# X['Sec4_CardingLevel'] = X['Sec4_CardingLevel'].astype(int)
# y = df_DNN.iloc[:, df_DNN.columns.get_loc("English_S4")]

# # Scale continuous variable in the dataset
# X['TScore'] = scaler.fit_transform(X['TScore'].values.reshape(-1, 1))
# X['English_S1'] = scaler.fit_transform(X['English_S1'].values.reshape(-1, 1))
# X['English_S2'] = scaler.fit_transform(X['English_S2'].values.reshape(-1, 1))
# X['English_S3'] = scaler.fit_transform(X['English_S3'].values.reshape(-1, 1))
# # y = scaler.fit_transform(y.values.reshape(-1, 1))

# # Split the Dataset into random Train and Test
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=2)
# # combined_df['EL'].value_counts()

# In[ ]:


# Library to rescale variables into the range [0,1]
from sklearn.preprocessing import MinMaxScaler

# Supress warning for SettingWithCopyError
pd.options.mode.chained_assignment = None

scaler = MinMaxScaler()

# Dataset
df_DNN = df_psych_original.copy().dropna()
# Extract Response and Predictors
X = df_DNN.drop(columns=['English_S4'], inplace=False)
# Convert Boolean to Integer (0,1)
X['Sec4_CardingLevel'] = X['Sec4_CardingLevel'].astype(int)
y = df_DNN.iloc[:, df_DNN.columns.get_loc("English_S4")]

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

Race_vocab = ['CHINESE', 'MALAY', 'INDIAN', 'OTHER', 'EURASIAN']
Race_column = tf.feature_column.categorical_column_with_vocabulary_list(
    key="Race", vocabulary_list=Race_vocab)

EL_vocab = ['A*', 'A', 'B', 'C', 'D']
EL_column = tf.feature_column.categorical_column_with_vocabulary_list(
    key="EL", vocabulary_list=EL_vocab)


Academy_vocab = ['Track & Field', 'Football', 'Netball', 'Shooting', 'Fencing', 'Bowling', 'Swimming', 'Badminton', 'Table Tennis',
                 'Individual Programme - Silat', 'Individual Programme - Golf', 'Individual Programme - Gymnastics',
                 'Individual Programme Aquatics – Water Polo', 'Individual Programme - Wushu', 'Individual Programme - Tennis',
                 'Individual Programme Aquatics - Artistic Swimming', 'Individual Programme - Indoor Sky Diving',
                 'Individual Programme - Triathlon', 'Individual Programme - Skating', 'Individual Programme - Cycling',
                 'Individual Programme - Obstacle Course Racing']
Academy_column = tf.feature_column.categorical_column_with_vocabulary_list(
    key="Academy", vocabulary_list=Academy_vocab)

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
    tf.feature_column.indicator_column(Race_column),
    tf.feature_column.indicator_column(Academy_column),
    tf.feature_column.indicator_column(Course_column),
    tf.feature_column.indicator_column(ResidentialType_column),
    tf.feature_column.indicator_column(EL_column),
    tf.feature_column.numeric_column(key='TScore', dtype=tf.float64),
    tf.feature_column.numeric_column(key='English_S1', dtype=tf.float64),
    tf.feature_column.numeric_column(key='English_S2', dtype=tf.float64),
    tf.feature_column.numeric_column(key='English_S3', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Sec4_CardingLevel', dtype=tf.int64),
    ###################
    tf.feature_column.numeric_column(key='Percentage', dtype=tf.float64),    
    tf.feature_column.numeric_column(key='Percentage1', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Percentage2', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Percentage3', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Percentage4', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Percentage5', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q1_M', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q1F', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q2', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q3', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q4', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q5', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q6', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q7', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q8a', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q8b', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q8c', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q1_6', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q2a', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q2b', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q2c', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q2d', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q2e', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q2f', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q2g', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q2h', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q2i', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q2j', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q2k', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q3_7', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q4a', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q4b', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q4c', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q4d', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q4e', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q4f', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q4g', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q4h', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q4i', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q4j', dtype=tf.float64),
    tf.feature_column.numeric_column(key='Q4k', dtype=tf.float64),
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
#     Adam as an optimizer (learning rate = 0.01)
#     Relu as an activation function
# 

# In[ ]:


model = tf.estimator.DNNRegressor(hidden_units=[32, 32, 32], feature_columns=feature_columns,
                                  optimizer=tf.optimizers.Adam(
                                      learning_rate=0.01),
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

print(final_preds)

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

# In[ ]:


mlp_data = combined_df.copy()

# To convert all these data into numerical value for decision tree
# Categorical data to scoring system
mlp_data['Gender'] = getGender(combined_df)
mlp_data['Sec4_CardingLevel'] = getCarding(combined_df)
mlp_data['Course'] = getcourse_tier(combined_df)
mlp_data['ResidentialType'] = gethousing(combined_df)
# Numerical data to scoring system
mlp_data['EL'] = encode_psle(combined_df, 'EL')
# mlp_data['English_S4'] = getresults(combined_df, 'English_S4')
mlp_data = mlp_data[mlp_data['English_S4'].notna()]
mlp_data = mlp_data[mlp_data['English_S3'].notna()]
mlp_data = mlp_data[mlp_data['English_S2'].notna()]
mlp_data = mlp_data[mlp_data['English_S1'].notna()]
# Columns that are applying KNN Imputer
cols = mlp_data.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
mlp_data = KNN_Imputer(mlp_data, cols)
mlp_data = pd.DataFrame(mlp_data, columns=cols)

# Validation
print(mlp_data.shape)


# In[ ]:


# to ignore if want to include express and NA
mlp_data = mlp_data[mlp_data['Course'] == 1]
print(mlp_data.shape)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# scaler.fit_transform()  # scales data
# scaler.inverse_transform()  # transforms scaled data back into original form
X = mlp_data[['TScore', 'Gender', 'Sec4_CardingLevel',
              'Course', 'EL', 'ResidentialType', 'English_S3', 'English_S2', 'English_S1']]
y = mlp_data[['English_S4']]
X = X.to_numpy()
y = y.to_numpy()

X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=2)


# In[ ]:


from sklearn.neural_network import MLPRegressor
clf = MLPRegressor(alpha=0.01, hidden_layer_sizes=(100, 20, 5), max_iter=1000,
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

# # Dataset
# df_combined_BLSTM = df_decisionTree_combined.copy()
# # df_combined_BLSTM.head(10)
# # Extract Response and Predictors
# X = df_combined_BLSTM.drop(columns=['English_S4'])
# X.drop(X.columns[9:],axis=1,inplace=True)

# # Grades with banding
# X['English_S3'] = getresults(df_combined_BLSTM, 'English_S3')
# X['English_S2'] = getresults(df_combined_BLSTM, 'English_S2')
# X['English_S1'] = getresults(df_combined_BLSTM, 'English_S1')
# df_combined_BLSTM['English_S4'] = getresults(df_combined_BLSTM, 'English_S4')
# y = df_combined_BLSTM['English_S4']
# count = 0
# for ele in X.columns:
#     print(count, ele)
#     count += 1

# # # Scale continuous variable in the dataset
# X = scaler.fit_transform(X)

# # Split the Dataset into random Train and Test
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=2)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# # Reshaping for nodes
# X_train = np.array(X_train.reshape(-1, 442, 9))
# X_test = np.array(X_test.reshape(-1, 111, 9))
# y_train = np.array(y_train.values.reshape(-1, 442, 1))
# y_test = np.array(y_test.values.reshape(-1, 111, 1))

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

# Supress warning for SettingWithCopyError
pd.options.mode.chained_assignment = None

# Library to rescale variables into the range [0,1]
scaler = MinMaxScaler()

# Dataset
df_combined_BLSTM = df_psych.copy()
# df_combined_BLSTM.head(10)
# Extract Response and Predictors
X = df_combined_BLSTM.drop(columns=['English_S4'])
X.drop(X.columns[50:],axis=1,inplace=True)
count = 0
for ele in X.columns:
    print(count, ele)
    count += 1
# Grades with banding
X['English_S3'] = getresults(df_combined_BLSTM, 'English_S3')
X['English_S2'] = getresults(df_combined_BLSTM, 'English_S2')
X['English_S1'] = getresults(df_combined_BLSTM, 'English_S1')
df_combined_BLSTM['English_S4'] = getresults(df_combined_BLSTM, 'English_S4')
y = df_combined_BLSTM['English_S4']
# X.head()

# Scale continuous variable in the dataset
X = scaler.fit_transform(X)

# Split the Dataset into random Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Reshaping for nodes
X_train = np.array(X_train.reshape(-1, 85, 50))
X_test = np.array(X_test.reshape(-1, 22, 50))
y_train = np.array(y_train.values.reshape(-1, 85, 1))
y_test = np.array(y_test.values.reshape(-1, 22, 1))

# In[ ]:


# Feature Scaling
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout, Bidirectional

regressor = Sequential()
regressor.add(LSTM(units=20, activation='sigmoid',
              input_shape=(None, 50), return_sequences=True)) # Tweak the parameter to match the column size
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
spare = predicted.copy()


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
    if abs(y_test[i] - predicted[i].round()) <= 1:
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


xg_el_num = combined_df.copy()

# To convert all these data into numerical value for decision tree
# Categorical data to scoring system
xg_el_num['Gender'] = getGender(combined_df)
xg_el_num['Sec4_CardingLevel'] = getCarding(combined_df)
xg_el_num['Course'] = getcourse_tier(combined_df)
xg_el_num['ResidentialType'] = gethousing(combined_df)
# Numerical data to scoring system
xg_el_num['EL'] = encode_psle(combined_df, 'EL')
xg_el_num = xg_el_num[xg_el_num['English_S4'].notna()]
xg_el_num = xg_el_num[xg_el_num['English_S3'].notna()]
xg_el_num = xg_el_num[xg_el_num['English_S2'].notna()]
xg_el_num = xg_el_num[xg_el_num['English_S1'].notna()]
# Columns that are applying KNN Imputer
cols = xg_el_num.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
xg_el_num = KNN_Imputer(xg_el_num, cols)
xg_el_num = pd.DataFrame(xg_el_num, columns=cols)

# Validation
print(xg_el_num.shape)


# In[ ]:


# to ignore if want to include express and NA
xg_el_num = xg_el_num[xg_el_num['Course'] == 1]
print(xg_el_num.shape)


# In[ ]:


scaler = MinMaxScaler()
# scaler.fit_transform()  # scales data
# scaler.inverse_transform()  # transforms scaled data back into original form
X = xg_el_num[['TScore', 'Gender', 'Sec4_CardingLevel',
               'Course', 'EL', 'ResidentialType', 'English_S3', 'English_S2', 'English_S1']]
y = xg_el_num[['English_S4']]
X = X.to_numpy()
y = y.to_numpy()

X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.3, random_state=2)
y_test_final = scaler.inverse_transform(y_test).flatten()

xgr = XGBRegressor(learning_rate = 0.32, max_depth = 3)
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


# Make a duplicate dataframe for manipulation
df_test = df_psych_original.copy()

# To convert all these data into numerical value for decision tree
# Categorical data to scoring system
df_test['Gender'] = getGender(df_psych_original)
df_test['Sec4_CardingLevel'] = getCarding(df_psych_original)
df_test['Course'] = getcourse_tier(df_psych_original)
df_test['ResidentialType'] = gethousing(df_psych_original)

# Numerical data to scoring system
df_test['EL'] = encode_psle(df_psych_original, 'EL')
df_test = df_test[df_test['English_S4'].notna()]
df_test = df_test[df_test['English_S3'].notna()]
df_test = df_test[df_test['English_S2'].notna()]
df_test = df_test[df_test['English_S1'].notna()]

# Columns that are applying KNN Imputer
cols = df_test.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
df_test = KNN_Imputer(df_test, cols)
df_test = pd.DataFrame(df_test, columns=cols)

# Reset the df index
df_test.reset_index(inplace=True)

# Restoring the 2 un-affected columns
df_test['Race'] = df_psych_original['Race'].copy()
df_test['Academy'] = df_psych_original['Academy'].copy()

# Validation
print(df_test.shape)
df_test.head(5)
df_test1 = pd.get_dummies(df_test, columns=[
    'Race', 'Academy'], drop_first=True)

# In[ ]:


# to ignore if want to include express and NA
# df_test1 = df_test1[df_test1['Course'] == 1]
# print(df_test1.shape)

# In[ ]:


from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler()

X = df_test1.copy()
print(X.shape)
X.drop(columns=['English_S4'], inplace=True)
# for col in X.columns:
#     print(col)
y = df_test1[['English_S4']]

X = X.to_numpy()
y = y.to_numpy()

X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=2)
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


# Make a duplicate dataframe for manipulation
df_test = df_psych_original.copy()

# To convert all these data into numerical value for decision tree
# Categorical data to scoring system
df_test['Gender'] = getGender(df_psych_original)
df_test['Sec4_CardingLevel'] = getCarding(df_psych_original)
df_test['Course'] = getcourse_tier(df_psych_original)
df_test['ResidentialType'] = gethousing(df_psych_original)

# Numerical data to scoring system
df_test['EL'] = encode_psle(df_psych_original, 'EL')
# df_test = df_test[df_test['English_S4'].notna()]
# df_test = df_test[df_test['English_S3'].notna()]
# df_test = df_test[df_test['English_S2'].notna()]
# df_test = df_test[df_test['English_S1'].notna()]

# Columns that are applying KNN Imputer
cols = df_test.columns.to_numpy()
cols = np.delete(cols, np.argwhere(cols == 'Race'))
cols = np.delete(cols, np.argwhere(cols == 'Academy'))

# Fill up the NaN based on KNN Imputer
df_test = KNN_Imputer(df_test, cols)
df_test = pd.DataFrame(df_test, columns=cols)

# Reset the df index
df_test.reset_index(inplace=True)

# Restoring the 2 un-affected columns
df_test['Race'] = df_psych_original['Race'].copy()
df_test['Academy'] = df_psych_original['Academy'].copy()

# Validation
print(df_test.shape)
df_test.head(5)
df_test1 = pd.get_dummies(df_test, columns=[
    'Race', 'Academy'], drop_first=True)

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler()

X = df_test1.copy()
print(X.shape)
X.drop(columns=['English_S4'], inplace=True)
# for col in X.columns:
#     print(col)
y = df_test1[['English_S4']]

X = X.to_numpy()
y = y.to_numpy()

X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=2)
y_test_final = scaler.inverse_transform(y_test).flatten()

# In[ ]:


from sklearn.neural_network import MLPRegressor
clf = MLPRegressor(alpha=0.01, hidden_layer_sizes=(100, 20, 5), max_iter=1000,
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
