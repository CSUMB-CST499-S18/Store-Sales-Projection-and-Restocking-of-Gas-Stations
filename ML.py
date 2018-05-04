import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from statsmodels.tsa.arima_process import arma_generate_sample
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import lag_plot
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.graphics.tsaplots import plot_pacf
# from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
from pandas import DataFrame
from pandas import concat
from sklearn import linear_model
from sklearn import svm

#import data
dat = pd.read_csv("./data/sales.csv")

#format Date so that it does not contain hours and set it as index
dat['Date'] = dat['Date'].str[:10]
dat.set_index('Date')

#drop information that is not needed
df = dat.drop(['POSCode', 'SalesAmount'], axis = 1)

#In the future the item in question will be passed in instead of hard coded.
item = df.loc[df['Description'] == 'DEW  20OZ']
item = item.drop(['Description'], axis = 1)

#Convert the date to proper datetime format
item['Date'] = pd.to_datetime(item['Date'],format='%Y-%m-%d')

#add days of the week
item['weekday'] = item['Date'].dt.dayofweek
# print(item)

#flip dataframe upside down
item = item.iloc[::-1]
#one hot encode days
item= pd.get_dummies(item, columns=['weekday'], prefix=['weekday'])

item = item.set_index('Date')

#MACHINE LEARNING ----------------------------------------

#get the dates
dates = item.index.values

#keep the values from the dataframe
X = item.values

#Function that turns the time series into a supervised problem
def make_supervised(item):
    df = DataFrame(item)

    columns = [df.shift(i) for i in range(1,2)]
    columns.append(df)

    df = concat(columns,axis=1)

    df.fillna(0, inplace=True)
    print(df)
    return(df)

data = make_supervised(X)

#select inputs and labels
X = data.iloc[:,0:8]
y = data.iloc[:,8]

#Break into training and testing data
X_train = X[:385]
X_test = X[385:]

# X_test
y_train = y[:385]
y_test = y[385:]
predicted_dates = dates[385:]

#ridge regression
clf1 = linear_model.Ridge(alpha=2)
clf1.fit(X_train,y_train)

#suppor vector machine regression
clf2 = svm.SVR()
clf2.fit(X_train,y_train)

predicts_rrg = clf1.predict(X_test)
predicts_svm = clf2.predict(X_test)

print('{')
for i in range(10):
    print('"' + str(predicted_dates[i]) + '":', predicts_rrg[i], ',')
print('}')

print("Ridge Error", np.sqrt(mean_squared_error(predicts_rrg,y_test)))
print("SVM Error", np.sqrt(mean_squared_error(predicts_svm,y_test)))
