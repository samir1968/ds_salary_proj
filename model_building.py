# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:23:59 2021

@author: samir
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("eda_data.csv")

# choose relevant columns
print(df.columns)

df_model = df[['avg_salary', 'Rating', 'Size', 'Type of ownership',
               'Industry', 'Sector', 'Revenue', 'num_comp','hourly',
               'employer provided salary', 'job_state', 'same_location',
               'age', 'python_yn', 'spark', 'aws', 'excel', 'job_simp', 
               'seniority', 'descr_len' ]]
# get dummy data
df_dum = pd.get_dummies(df_model)

# train test split
X = df_dum.drop('avg_salary', axis=1)
y = df_dum.avg_salary.values
print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# multiple linear regression

import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
print(model.fit().summary())

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

r_cv_score = np.mean(cross_val_score(lm, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))
print("\n Linear Regression Score: ",r_cv_score)

# lasso regression

lm_l = Lasso(alpha=0.13)
lm_l.fit(X_train, y_train)
l_cv_score = np.mean(cross_val_score(lm_l, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))
print("\n Lasso Regression Score: ",l_cv_score)

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))

plt.plot(alpha, error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns=['alpha', 'error'])
min_err = df_err[df_err.error == max(df_err.error)]


# random forest

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
print("\n RandomForest Score: ",np.mean(cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))

# tune models GridsearchCV

from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,100,10),'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}
gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error',cv=3)
gs.fit(X_train, y_train)
print("\n GridSearchCV Best_Score: ",gs.best_score_)
print(gs.best_estimator_)
# test ensembles

tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test,tpred_lm))
print(mean_absolute_error(y_test,tpred_lml))
print(mean_absolute_error(y_test,tpred_rf))

print("\n Score of Linear Regression And RandomForest: ",mean_absolute_error(y_test,(tpred_lm + tpred_rf)/2))