import wget as wget
import ssl

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# data download
# ssl._create_default_https_context = ssl._create_unverified_context
# wget.download("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv")

# 1 EDA
df = pd.read_csv('housing.csv')
print(df.head())
print(df.columns)
print(df.dtypes)

columns_needed = [

'latitude',
'longitude',
'housing_median_age',
'total_rooms',
'total_bedrooms',
'population',
'households',
'median_income',
'median_house_value',
'ocean_proximity'
]

df = df[columns_needed]
print(df.head())
print(df.dtypes)


# 2 Data cleaning

# adding new columns

df['rooms_per_household'] = df.total_rooms / df.households
df['bedrooms_per_room'] = df.total_bedrooms / df.total_rooms
df['population_per_household'] = df.population / df.households
print(df.head)

def fill_with_zeros(df):
    df = df.copy()
    df = df.fillna(0)
    return df

#filling missing values with 0
df_cleaned = fill_with_zeros(df)

print(df_cleaned)

# Question 1

ocean_proximity_mode = df_cleaned['ocean_proximity'].mode()
print('most frequent value of "ocean_proximity" column in the dataset', ocean_proximity_mode)

# Question 2 Correlation matrix

corrMatrix = df_cleaned.corr()
# print(corrMatrix)
sns.heatmap(corrMatrix, annot=True, xticklabels=True, yticklabels=True)
plt.show()

#answer: total_bedrooms and households

# Make median_house_value binary
df_cleaned['median_house_value'] = np.where(df['median_house_value'] > df['median_house_value'].mean(), 1, 0)

print(df_cleaned.head())

#  Split the data

def get_values(df):
    X = df.values
    return X


X_train, X_test, y_train, y_test = train_test_split(get_values(df), get_values(df.median_house_value), test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
