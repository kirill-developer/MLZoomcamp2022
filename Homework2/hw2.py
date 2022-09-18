import wget as wget
import ssl

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

#data download
# ssl._create_default_https_context = ssl._create_unverified_context
# wget.download("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv")

#1 EDA
df = pd.read_csv('housing.csv')
print(df.head())

sns.histplot(df.median_house_value, bins=100)
plt.show()

#distribution has a long tail which it typical for price distributions


#zooming in
sns.histplot(df.median_house_value[df.median_house_value < 300000], bins=100)
plt.show()

#tail is not good for ML because it will confuse dodel
#we will apply log distributuon to fix it
# we cant have log[0], because its not exist.
# To fix it we will just add 1 everywhere (log1p in pandas)

price_logs = np.log1p(df.median_house_value)

print(price_logs)

sns.histplot(price_logs, bins=100)
plt.show()

#normal distribution shape.
# Models work way better with normal distribution
#than with tail ones



print(df.columns)
print(df.dtypes)

columns_needed = [

'latitude','longitude',
'housing_median_age',
'total_rooms',
'total_bedrooms',
'population',
'households',
'median_income',
'median_house_value'

]

df = df[columns_needed]
print(df.head())
print(df.dtypes)

#Question 1 - Find a feature with missing values. How many missing values does it have?

print(df.isnull().sum())

# answer: total_bedrooms        207

#Question 2 - What's the median (50% percentile) for variable 'population'?

print(df['population'].describe())
print('median value of population column in the dataset', df['population']. median())

# answer:        1166

#2 data split

n = len(df)
n_val = int(n*0.2)
n_test = int(n*0.2)
n_train = n-n_val-n_test

#check for consistency
print(n, n_val + n_test + n_train)
print(n_val, n_test, n_train)


#shuffle to solve sequential problem
idx = np.arange(n)
#to make results reproducible we need to use numpy seed = 42
np.random.seed(42)
np.random.shuffle(idx)
print(df.iloc[idx[:10]])


df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]

print(df_train)

print(len(df_train), len(df_val), len(df_test))

#index reset
df_train = df_train.reset_index(drop=True)
# print(df_train)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

#perform log transformation to y
#getting into numpy series
y_train = np.log1p(df_train.median_house_value.values)
print(y_train)
y_val = np.log1p(df_val.median_house_value.values)
y_test = np.log1p(df_test.median_house_value.values)

#remove msrp variable to avoid accidental usage! Good learining!
# to avoid usage price variable as a feature in predicting price,
# then perfect model but wrong one

#split data -> isolate target values in variables -> delete it from df completely
del df_train['median_house_value']
del df_val['median_house_value']
del df_test['median_house_value']

print(len(y_train))

#question 3

#filling missing values in total_bedrooms with 0

def prepare_X_fill_with_zeros(df):
    df = df.copy()
    df = df.fillna(0)
    X = df.values
    return X

def prepare_X_fill_with_mean(df):
    df = df.copy()
    df = df.fillna(df['total_bedrooms'].mean())
    X = df.values
    return X

# print(prepare_X_fill_with_mean(df_train))


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]

def rmse(y_pred, y):
    se = (y_pred - y)**2
    mse = se.mean()
    return np.sqrt(mse)

# training and validating for opption 1: empty values filled with 0
#training part
X_train = prepare_X_fill_with_zeros(df_train)
w0, w = train_linear_regression(X_train, y_train)

#validation part
X_val = prepare_X_fill_with_zeros(df_val)
y_pred = w0 + X_val.dot(w)

print('RMSE for option with empty values filled with 0: ', round(rmse(y_pred, y_val), 2))

# training and validating for option 2: empty values filled with mean of the variable 'total_bedrooms'
#training part
X_train = prepare_X_fill_with_mean(df_train)
w0, w = train_linear_regression(X_train, y_train)

#validation part
X_val = prepare_X_fill_with_mean(df_val)
y_pred = w0 + X_val.dot(w)

print('RMSE for option with empty values filled with mean: ', round(rmse(y_pred, y_val), 2))

#answer: Both are equally good

#Question 4  train a regularized linear regression

def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]

# we will use validation set for finding the best value of r

for r in [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
    # training part
    X_train = prepare_X_fill_with_zeros(df_train)
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)

    # validation part
    X_val = prepare_X_fill_with_zeros(df_val)
    y_pred = w0 + X_val.dot(w)

    score = rmse(y_pred, y_val)

    print(r, w0, round(score, 2))


# answer: smallest r is 0

#Question 5 use diff seed values





RMSE_set = []

for s in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:

    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    # check for consistency
    # print(n, n_val + n_test + n_train)
    # print(n_val, n_test, n_train)

    # shuffle to solve sequential problem
    idx = np.arange(n)
    # to make results reproducible we need to use numpy seed = 42
    np.random.seed(s)
    np.random.shuffle(idx)
    # print(df.iloc[idx[:10]])

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_train + n_val]]
    df_test = df.iloc[idx[n_train + n_val:]]

    # print(df_train)

    # print(len(df_train), len(df_val), len(df_test))

    # index reset
    df_train = df_train.reset_index(drop=True)
    # print(df_train)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # perform log transformation to y
    # getting into numpy series
    y_train = np.log1p(df_train.median_house_value.values)
    # print(y_train)
    y_val = np.log1p(df_val.median_house_value.values)
    y_test = np.log1p(df_test.median_house_value.values)

    # remove msrp variable to avoid accidental usage! Good learining!
    # to avoid usage price variable as a feature in predicting price,
    # then perfect model but wrong one

    # split data -> isolate target values in variables -> delete it from df completely
    del df_train['median_house_value']
    del df_val['median_house_value']
    del df_test['median_house_value']

    # print(len(y_train))
    # training part
    X_train = prepare_X_fill_with_zeros(df_train)
    w0, w = train_linear_regression(X_train, y_train)
    # validation part
    X_val = prepare_X_fill_with_zeros(df_val)
    y_pred = w0 + X_val.dot(w)

    rmse_result = rmse(y_pred, y_val)
    print('RMSE for seed = %s:' % s, rmse_result)
    RMSE_set.append(rmse_result)

print(RMSE_set)
print(len(RMSE_set))

std = np.std(RMSE_set)
print('standard deviation of all the scores', std)

# answer: 0.005 is the closest

#question 6:

n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test

# check for consistency
# print(n, n_val + n_test + n_train)
# print(n_val, n_test, n_train)

# shuffle to solve sequential problem
idx = np.arange(n)
# to make results reproducible we need to use numpy seed = 42
np.random.seed(9)
np.random.shuffle(idx)
# print(df.iloc[idx[:10]])

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val:]]

# print(df_train)

# print(len(df_train), len(df_val), len(df_test))

# index reset
df_train = df_train.reset_index(drop=True)
# print(df_train)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# perform log transformation to y
# getting into numpy series
y_train = np.log1p(df_train.median_house_value.values)
# print(y_train)
y_val = np.log1p(df_val.median_house_value.values)
y_test = np.log1p(df_test.median_house_value.values)

# remove target variable to avoid accidental usage! Good learining!
# to avoid usage price variable as a feature in predicting price,
# then perfect model but wrong one

# split data -> isolate target values in variables -> delete it from df completely
del df_train['median_house_value']
del df_val['median_house_value']
del df_test['median_house_value']

df_full_train = pd.concat([df_train, df_val])

print(df_full_train)

df_full_train = df_full_train.reset_index(drop=True)

print(df_full_train)

X_full_train = prepare_X_fill_with_zeros(df_full_train)
print(X_full_train)

y_full_train = np.concatenate([y_train, y_val])

w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)

#prepare testing DS
X_test = prepare_X_fill_with_zeros(df_test)

y_pred = w0 + X_test.dot(w)

score = rmse(y_test, y_pred)

print(round(score, 3))

#answer: 0.35


