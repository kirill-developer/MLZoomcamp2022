import numpy as np
import pandas as pd


import ssl
import seaborn as sns

import matplotlib as plt

print(np.__version__)

ssl._create_default_https_context = ssl._create_unverified_context
url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv'

df = pd.read_csv(url)

print(df.head())

print(df.describe())

print('Rows count', len(df.index))

# print('3 most frequent manufacturers', df.Make.mode())

# get top 3 most frequent car brands
n = 3
print('3 most frequent manufacturers', df['Make'].value_counts()[:n].index.tolist())

# get unique audi models
print('Unique Audi models',
      df.query('Make == "Audi"')['Model'].agg(['nunique','count','size']))

#Counting rows with missing values / can be done better!

print('Counting rows with missing values', df.isnull().sum(axis = 0))

#pseudocode
#iterate over rows in df above and if value >0 increase count
#print count

#Question 6
print('median value of "Engine Cylinders" column in the dataset', df['Engine Cylinders']. median())
cylinders_mode = df['Engine Cylinders'].mode()
print('most frequent value of "Engine Cylinders" column in the dataset', cylinders_mode)

values = {"Engine Cylinders": cylinders_mode}

df = df.fillna(values)

cylinders_median_updated = df['Engine Cylinders']. median()
print('updated median value of "Engine Cylinders" column in the dataset', cylinders_median_updated)

#Question 7

df_lotus = df.loc[df['Make'] == 'Lotus']

print(df_lotus)

df_lotus_filtered = df_lotus[['Engine HP','Engine Cylinders']]

print(df_lotus_filtered)

df_clean = df_lotus_filtered.drop_duplicates(subset=['Engine HP','Engine Cylinders'])

print(df_clean)
print('length after duplicate removal', len(df_clean))

X = df_clean.to_numpy()

print(X)

XTX = np.dot(X.T, X)
print(XTX)

print(np.linalg.det(XTX))
#if The determinant of our matrix is zero, then we run into an error LinAlgError: Singular matrix

#Invert XTX
XTX_inv = np.linalg.inv(XTX)
print(XTX_inv)

y = np.array([1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800])

print(y)

# Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w

w1 = np.dot(XTX_inv, X.T)
w = np.dot(w1, y)

print('w= ', w)