import wget as wget
import ssl

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  roc_auc_score
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
from sklearn.metrics import auc

# data download
# ssl._create_default_https_context = ssl._create_unverified_context
# wget.download("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/AER_credit_card_data.csv")

# Preparation
#The goal of this homework is to inspect the output of different evaluation metrics by creating a classification model (target column card).

df = pd.read_csv('AER_credit_card_data.csv')
print(df.head())
#see all columns
print(df.head().T)
print(df.dtypes)

# Create the target variable by mapping yes to 1 and no to 0
df.card = pd.Series(np.where(df.card.values == 'yes', 1, 0), df.index)
print(df.head())

# Split the dataset into 3 parts: train/validation/test with 60%/20%/20% distribution. Use train_test_split funciton for that with random_state=1.
# Setting validation framework

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

print(len(df_full_train), len(df_test))

df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

print(len(df_train), len(df_val), len(df_test))

#resetting indexes
df_train = df_train.reset_index(drop=True)
# print(df_train.head())
df_val = df_val.reset_index(drop=True)
# print(df_val.head())
df_test = df_test.reset_index(drop=True)
# print(df_test.head())

#getting y values

y_train = df_train.card.values
y_val = df_val.card.values
y_test = df_test.card.values

#deleting y values from df

del df_train['card']
del df_val['card']
del df_test['card']

# Question 1


# plt.plot(df_train.index, df_train.reports, label='reports')
# plt.plot(df_train.index, df_train.dependents, label='dependents')
# plt.plot(df_train.index, df_train.active, label='active')
# plt.plot(df_train.index, df_train.share, label='share')
# plt.legend()
# plt.show()


auc_reports = roc_auc_score(y_train, -df_train.reports)
print('auc_reports', auc_reports)
auc_dependents = roc_auc_score(y_train, df_train.dependents)
print('auc_dependents', auc_dependents)
auc_active = roc_auc_score(y_train, df_train.active)
print('auc_active', auc_active)
auc_share = roc_auc_score(y_train, df_train.share)
print('auc_share', auc_share)

# answer: share

#???

# Question 2
columns_needed = ["reports", "age", "income", "share", "expenditure", "dependents", "months", "majorcards", "active", "owner", "selfemp"]

# one-hot encoding
train_dicts = df_train[columns_needed].to_dict(orient='records')
# print(train_dicts)
# print(df_train[['gender', 'contract']].iloc[:10])
# print(df_train[['gender', 'contract']].iloc[:10].to_dict(orient='records'))

#creating new instance of Dict Vectorizer class
dv = DictVectorizer(sparse=False)
# dv.fit(train_dicts)
X_train = dv.fit_transform(train_dicts)
# print(dv.get_feature_names())
# print(X_train.shape)

#repeating for validation DS
val_dicts = df_val[columns_needed].to_dict(orient='records')
# print(df_train[['gender', 'contract']].iloc[:10])
# print(df_train[['gender', 'contract']].iloc[:10].to_dict(orient='records'))
X_val = dv.transform(val_dicts)
# print(X_val.shape)

#training model
def train(df_train,y_train, C=1.0):
    dicts = df_train[columns_needed].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

dv, model = train(df_train, y_train, C=1.0)

#making the prediction
def predict(df, dv, model):
    dicts = df[columns_needed].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

y_pred = predict(df_val, dv, model)

#shortcut auc
print(roc_auc_score(y_val, y_pred).round(3))

#answer 0.995

#question 3
# p = tp / (tp + fp)
# print(p)
#
# r = tp / (tp + fn)
# print(r)

scores = []
tresholds =np.linspace(0, 1, 100)

for t in tresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)

    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()
    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()

    p = tp / (tp + fp)
    r = tp / (tp + fn)

    scores.append((t, tp, fp, fn, tn, p, r))
    # print(scores)

columns = ['treshold', 'tp', 'fp', 'fn', 'tn', 'p', 'r']
df_scores = pd.DataFrame(scores, columns=columns)
print(df_scores.head())

print(df_scores[::10])

#plotting

plt.plot(df_scores.treshold, df_scores.p, label='P')
plt.plot(df_scores.treshold, df_scores.r, label='R')
plt.legend()
plt.show()

#answer 0.3

# Question 4

# F1 = 2 * P * R / (P + R)

scores = []
tresholds =np.linspace(0, 1, 100)

for t in tresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)

    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()
    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()

    p = tp / (tp + fp)
    r = tp / (tp + fn)

    F1 = 2 * p * r / (p + r)

    scores.append((t, tp, fp, fn, tn, p, r, F1))
    # print(scores)

columns = ['treshold', 'tp', 'fp', 'fn', 'tn', 'p', 'r', 'F1']
df_scores = pd.DataFrame(scores, columns=columns)
print(df_scores.head())

print(df_scores[::10])

#plotting F1

plt.plot(df_scores.treshold, df_scores.F1, label='F1')
plt.legend()
plt.show()

#answer 0.4

# Question 5

scores = []

kfold = KFold(n_splits=10, shuffle=True, random_state=1)

for train_idx, val_idx in tqdm(kfold.split(df_full_train)):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.card.values
    y_val = df_val.card.values

    dv, model = train(df_train, y_train, C=1.0)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)

    scores.append(auc)

# print(len(train_idx))
# print(len(val_idx))
# print(len(df_full_train))

print(scores)

print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))

#answer 0.003

# Question 6

#finding optimal C

n_splits = 5

for C in tqdm([0.01, 0.1, 1, 10]):

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.card.values
        y_val = df_val.card.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)

        scores.append(auc)
        print()

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# answer 1 or 10. We can answer 1