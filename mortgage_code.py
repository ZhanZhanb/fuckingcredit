# %% 0. Settings
# imports

import pandas as pd
import numpy as np
import statsmodels.api as sm
import scorecardpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

pd.options.mode.chained_assignment = None

# %% 1. Data Import
data = pd.read_csv("mortgage_sample.csv")
print(data.count())
# pd.set_option('display.max_columns', None)
print(data.describe())

# remove empty values
df = data.dropna()

# plot outliers
df.plot()
print(df.count())
print(df.isnull().sum())
# remove outliers
# df = df.drop(df[['balance_orig_time'] > int(6e+06)].index)
df.drop(df[df['balance_orig_time'] > 2300000].index, inplace=True)
df.drop(df[df['LTV_time'] > 400].index, inplace=True)
df.drop(df[df['LTV_orig_time'] > 120].index, inplace=True)
df.drop(df[df['Interest_Rate_orig_time'] > 15].index, inplace=True)
df.drop(df[df['interest_rate_time'] > 20].index, inplace=True)
df.drop(df[df['Interest_Rate_orig_time'] == 0].index, inplace=True)
print(df.count())



# Resampling
# status0 = df.loc[df['status_time'] == 0]
# status1 = df.loc[df['status_time'] == 1]
# status2 = df.loc[df['status_time'] == 2]
#

#
# a_train, a_test= train_test_split(status0, test_size=0.3, random_state=0)
# b_train, b_test= train_test_split(status1, test_size=0.3, random_state=0)
# c_train, c_test= train_test_split(status2, test_size=0.3, random_state=0)
#
# train_y = pd.concat((a_train.loc[:, a_train.columns=='status_time'],
#                     b_train.loc[:, b_train.columns=='status_time'],c_train.loc[:,c_train.columns == 'status_time']), axis=0)
#
# train_x = pd.concat((a_train.loc[:, a_train.columns!='status_time'],
#                     b_train.loc[:, b_train.columns!='status_time'],c_train.loc[:,c_train.columns != 'status_time']),axis=0)
#
# test_y = pd.concat((a_test.loc[:, a_test.columns=='status_time'],
#                     b_test.loc[:, b_test.columns=='status_time'],c_test.loc[:,c_test.columns == 'status_time']),axis=0)
#
# test_x = pd.concat((a_test.loc[:, a_test.columns!='status_time'],
#                     b_test.loc[:, b_test.columns!='status_time'],c_test.loc[:,c_test.columns != 'status_time']), axis=0)

df['TARGET'] = df['default_time']
X = df.loc[:,df.columns != 'status_time']
y = df.loc[:,df.columns == 'status_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %% 3. Predictor preparation

print(df['default_time'].value_counts())
predictors = ["FICO_orig_time", "LTV_orig_time", "hpi_orig_time", "gdp_time", "uer_time"]
bins = sc.woebin(df[predictors + ['TARGET']], y='TARGET')
woe = sc.woebin_ply(X_train, bins)

# %% 4. Estimate logistic regression
y = woe.loc[:, 'TARGET']
X = woe.loc[:, [f"{pred}_woe" for pred in predictors]]

# Modelling
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.intercept_)
print(lr.coef_)

# SM to get more detailed results
logit_mod = sm.Logit(endog=y, exog=X)
estimated_model = logit_mod.fit()
estimated_model.summary()

# %% 5. Model Assessment
pred = lr.predict_proba(X)[:, 1]
confusion_matrix = confusion_matrix(y_test, pred)
print(confusion_matrix)

#K-FOLD cross validation with 10 folds.
from  sklearn  import  model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

# %% 6. Create scorecard
# score ------
points = 500
odds = 1 / 51
pdo = 50

card = sc.scorecard(bins, lr, X.columns,
                    points0=points,
                    odds0=odds,
                    pdo=pdo
                    )
for c in card:
    print(card[c])

# credit score
score = sc.scorecard_ply(data, card)

# %%
# log-odds to PD
print(np.log(.0025 / (1 - .0025)))
print(np.log(.018 / (1 - .018)))
print(np.log(.5 / (1 - .5)))

points = 500
odds = 1 / 51
pdo = 50

prob_good = .1

# score from probability
score = points + (pdo / np.log(2)) * np.log(prob_good / (1 - prob_good))
print(f'Score from probability:  {round(score, 0)}')

# %%
