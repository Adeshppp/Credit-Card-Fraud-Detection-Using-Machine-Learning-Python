import os 
import warnings
warnings.filterwarnings('ignore')
# Packages related to data importing, manipulation, exploratory data analysis and data understanding
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
# text customization package
from termcolor import colored as cl 
# Packages related to data visualizaiton
import seaborn as sns
import matplotlib.pyplot as plt

# Setting plot sizes and type of plot
plt.rc("font", size=14)
plt.rcParams['axes.grid'] = True
plt.figure(figsize=(6,3))
plt.gray()
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.preprocessing import  PolynomialFeatures, KBinsDiscretizer, FunctionTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, OrdinalEncoder
import statsmodels.formula.api as smf
import statsmodels.tsa as tsa
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import *
from sklearn.ensemble import BaggingClassifier, BaggingRegressor,RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor 
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# importing data from csv file
data=pd.read_csv("creditcard.csv")

# distribution of transaction
Total_transactions = len(data)
normal = len(data[data.Class == 0])
fraudulent = len(data[data.Class == 1])
fraud_percentage = round(fraudulent/normal*100, 2)
print(cl('Total number of Trnsactions are {}'.format(Total_transactions), attrs = ['bold']))
print(cl('Number of Normal Transactions are {}'.format(normal), attrs = ['bold']))
print(cl('Number of fraudulent Transactions are {}'.format(fraudulent), attrs = ['bold']))
print(cl('Percentage of fraud Transactions is {}'.format(fraud_percentage), attrs = ['bold']))

data.info()

print('min and max of data amount ',min(data.Amount),max(data.Amount))

# using standard scaler
sc = StandardScaler()
amount = data['Amount'].values
data['Amount'] = sc.fit_transform(amount.reshape(-1, 1))

print('data shape ',data.shape)
print('dropping external deciding factor (Time)')

# dropping external deciding factor
data.drop(['Time'], axis=1, inplace=True)
print('data shape ',data.shape)
print('dropping duplicate transactions ')

# removing duplicates
data.drop_duplicates(inplace=True)
print('data shape ',data.shape)

#====================================================================================================
#============================= Train and Test Split ===================================

# dependent and independent variables
X = data.drop('Class', axis = 1).values
y = data['Class'].values

#  splitting the Train and Test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

# print(data.shape)
#====================================================================================================
#============================= Model Building ===================================


# 1. Decision Tree

DT = DecisionTreeClassifier(max_depth = 4, criterion = 'entropy')
DT.fit(X_train, y_train)
dt_yhat = DT.predict(X_test)

print('Accuracy score of the Decision Tree model is {}'.format(accuracy_score(y_test, dt_yhat)))

# checking F1 score 
print('F1 score of the Decision Tree model is {}'.format(f1_score(y_test, dt_yhat)))

# checking decision matrix
confusion_matrix(y_test, dt_yhat, labels = [0, 1])
print('Confusion Matrix of the Decision Tree model ')
print(confusion_matrix(y_test, dt_yhat, labels = [0, 1]))

#====================================================================================================

# 2. K-Nearest Neighbours

n = 7
KNN = KNeighborsClassifier(n_neighbors = n)
KNN.fit(X_train, y_train)
knn_yhat = KNN.predict(X_test)

# checking accuracy
print('Accuracy score of the K-Nearest Neighbors model is {}'.format(accuracy_score(y_test, knn_yhat)))

# checking F1 score
print('F1 score of the K-Nearest Neighbors model is {}'.format(f1_score(y_test, knn_yhat)))

# checking confusion matrix 
confusion_matrix(y_test, knn_yhat, labels = [0, 1])
print('Confusion Matrix of the  K-Nearest Neighbours ')
print(confusion_matrix(y_test, knn_yhat, labels = [0, 1]))


#====================================================================================================

# 3. Logistic Regression

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_yhat = lr.predict(X_test)

# checking accuracy
print('Accuracy score of the Logistic Regression model is {}'.format(accuracy_score(y_test, lr_yhat)))

# checking F1 score
print('F1 score of the Logistic Regression model is {}'.format(f1_score(y_test, lr_yhat)))


# checking confusion matrix 
confusion_matrix(y_test, lr_yhat, labels = [0, 1])
print('Confusion Matrix of the Logistic Regression ')
print(confusion_matrix(y_test, lr_yhat, labels = [0, 1]))






#====================================================================================================

# 5. Random Forest

rf = RandomForestClassifier(max_depth = 4)
rf.fit(X_train, y_train)
rf_yhat = rf.predict(X_test)

# checking accuracy
print('Accuracy score of the Random Forest model is {}'.format(accuracy_score(y_test, rf_yhat)))

# checking F1 score
print('F1 score of the Random Forest model is {}'.format(f1_score(y_test, rf_yhat)))


# checking confusion matrix 
confusion_matrix(y_test, rf_yhat, labels = [0, 1])
print('Confusion Matrix of the Random Forest ')
print(confusion_matrix(y_test, rf_yhat, labels = [0, 1]))



#====================================================================================================

# 6. XGBoost model

xgb = XGBClassifier(max_depth = 4)
xgb.fit(X_train, y_train)
xgb_yhat = xgb.predict(X_test)

# checking accuracy
print('Accuracy score of the XGBoost model is {}'.format(accuracy_score(y_test, xgb_yhat)))

# checking F1 score
print('F1 score of the XGBoost model is {}'.format(f1_score(y_test, xgb_yhat)))


# checking confusion matrix 
confusion_matrix(y_test, xgb_yhat, labels = [0, 1])
print('Confusion Matrix of the XGBoost model ')
print(confusion_matrix(y_test, xgb_yhat, labels = [0, 1]))
