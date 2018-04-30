import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from statsmodels.tsa.arima_process import arma_generate_sample
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import lag_plot
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error
import warnings
from pandas import DataFrame
from pandas import concat
import tensorflow as tf
from random import randint
from flask import Flask
from flask import jsonify

# from keras.layers.core import Dense, Activation, Dropout
# from keras.layers.recurrent import LSTM
# from keras.models import Sequential
import json


#top level function to return the results
def getResults(itemName):

    print("The item passed in is:", itemName['userInput'])

    itemName = itemName['userInput']
    
    #open the file
    df = openPreprocess("./data/sales.csv")

    itemSales = getSales(df, itemName)

    #get the dates
    dates = itemSales.index.values

    #keep the values from the dataframe
    X = itemSales.values
    # print(X)

    #turn the time series into supervised data
    data = makeSupervised(X)

    X = data.iloc[:,0:8]
    y = data.iloc[:,8]

    #Break into training and testing data
    X_train = X[:350]
    X_val = X[350:380]
    X_test = X[380:]

    y_train = y[:350]
    y_val = y[350:380]
    y_test = y[380:]
    predicted_dates = dates[380:]

    X_train2 = X[:380]
    y_train2 = y[:380]

    #train the neural network and return the results
    results, RMSE = neuralNet(X_train, y_train, X_val, y_val, X_test, y_test)
    # results2, RMSE2 = LSTMModel(X_train2, y_train2, X_test, y_test)

    print("The predictions of the next 10 days are: ")


    #Compare with currently used method
    print("The neural net performed: ", simpleAverageRMSE(RMSE, X_test, y_test), "% times better compared to the previous method!")
    # print("The ARIMA model performed: ", simpleAverageRMSE(RMSE2, X_test, y_test), "% times better compared to the previous method!")

    return predicted_dates, results



#opens CSV and preprocesses data
def openPreprocess(path):

    dat = pd.read_csv(path)

    #format Date so that it does not contain hours and set it as index
    dat['Date'] = dat['Date'].str[:10]
    dat.set_index('Date')

    #drop information that is not needed
    df = dat.drop(['POSCode', 'SalesAmount'], axis = 1)

    return df


#Selects the data for the item provided, and processes the data to create a time series
def getSales(df, itemName):
    item = df.loc[df['Description'] == itemName]
    # print(item)
    item = item.drop(['Description'], axis = 1)

    #Convert the date to proper datetime format
    item['Date'] = pd.to_datetime(item['Date'],format='%Y-%m-%d')

    #add days of the week
    item['weekday'] = item['Date'].dt.dayofweek
    # print(item)

    #flip dataframe upside down
    item = item.iloc[::-1]

    # print(item)
    item = item.set_index('Date')

    #Account for the days that the item was not sold by adding those dates
    idx = pd.date_range('2016-11-01', '2017-11-30')
    item = item.reindex(idx, fill_value=0)

    # print("Shape after accounting for dates with no sales", item.shape)

    #one hot encode days
    item= pd.get_dummies(item, columns=['weekday'], prefix=['weekday'])

    return item


#Turns the timeseries into a supervised machine learning problem with inputs
#and outputs.
def makeSupervised(item):
    df = DataFrame(item)

    #shift columns so that the input of one instance can be the output of the
    #previous one
    columns = [df.shift(i) for i in range(1,2)]
    columns.append(df)

    df = concat(columns,axis=1)

    df.fillna(0, inplace=True)

    return df

def simpleAverageRMSE(RMSE, X_val, y_test):
    last10 = X_val.loc[20:,0]

    average = np.mean(last10)

    #make 10 same ones
    predicts_avg = [average]*15

    # print("Simple average prediction RMSE: ", np.sqrt(mean_squared_error(predicts_avg,y_test)))
    return np.sqrt(mean_squared_error(predicts_avg,y_test))*100/RMSE

def arimaModel(X_train, y_train, X_val, y_val, X_test, y_test):

    X_tr = X_train.iloc[:,0].values
    X_v = X_val.iloc[:,0].values
    X_te = X_test.iloc[:,0].values

    print(X_tr)

    #suppress warnings
    warnings.filterwarnings("ignore")

    #tune the model and find the best parameters
    bestorder = arimaGridSearch(X_tr, X_v, [1,2,3,4,5],[0,1,2,3],[0,1,2,3,4,5])

    model = ARIMA(X_tr, bestorder)
    model_fit = model.fit()

    predict = model_fit.forecast(steps=forecast_length)[0]

    # compute the error
    rmse = np.sqrt(mean_squared_error(X_te, predict))

    print("ARIMA Test RMSE: ", rmse)

    return predictions, RMSE

#this dickey fuller test checks stationarity of the time series in order to fit an ARIMA model
def testStationarity(X):

    print("Results of Dickey-Fuller Test:")

    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


#Build and run the Neural Net
def neuralNet(X_train, y_train, X_val, y_val, X_test, y_test):

    #define number of features, and nodes in hidden layers
    n_inputs = 8  # MNIST
    n_hidden1 = 20
    n_hidden2 = 20
    n_hidden3 = 20
    n_hidden4 = 20
    #number of outputs is 1, since we have a regression task
    n_outputs = 1

    #learning rate of the optimizer. The optimal learning rate is different for
    #different tasks
    learning_rate = 0.0005

    tf.reset_default_graph()

    #Batches will be fed in the placeholders
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.float32, shape=(None), name="y")

    #Build hidden layers
    #elu activation function is used instead of relu
    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                                  activation=tf.nn.elu)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                                  activation=tf.nn.elu)
        # hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3",
        #                           activation=tf.nn.elu)
        # hidden4 = tf.layers.dense(hidden3, n_hidden4, name="hidden4",
        #                           activation=tf.nn.elu)
        results = tf.layers.dense(hidden2, n_outputs, name="outputs")

    #Calculate the results and the mean squared error
    with tf.name_scope("loss"):
        results = tf.squeeze(results) #squeeze is used to get rid of an extra dimension of 1
        mse = tf.losses.mean_squared_error(y,results)

    #Backpropagation
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate) #Adam performs better than normal Gradient Descent
        training_op = optimizer.minimize(mse)

    #initialize graph
    init = tf.global_variables_initializer()

    n_epochs = 300 #times the data will be trained on
    batch_size = 30
    min = 100 #minimum validation rmse
    count = 3


    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):

            #create batches
            for i in range(5):
                i = randint(0, 319)
                X_batch = X_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]
                sess.run(training_op, feed_dict={X: X_train, y: y_train})
            #Train on the training data

            #Test on the validation data every 5000 epochs
            if epoch % 20 == 0:


                currvalerror = np.sqrt(mse.eval(feed_dict={X: X_val, y: y_val}))
                if(currvalerror < min):
                    min = currvalerror
                    count = 2
                else:
                    count -= 1
                    if(count == 0):
                        break

                print("Epoch", epoch, "RMSE (lower is better) =", currvalerror)

        #Finally, test on the test dataset
        predictions = results.eval(feed_dict={X: X_test, y: y_test})
        testRMSE = np.sqrt(mse.eval(feed_dict={X: X_test, y: y_test}))

        print("NN Test RMSE: ", testRMSE)
        return predictions, testRMSE


def LSTMModel(X_train, y_train, X_test, y_test):

    #Keep the values in order to reshape
    X_train = X_train.values
    X_test = X_test.values


    # inputs need to be reshaped in the form of [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    #Sequentially append layers
    model = Sequential()

    #Add an LSTM layer
    model.add(LSTM(input_shape=(1,8), output_dim=20, return_sequences=True))

    #Add 20% probability of dropout in layer
    model.add(Dropout(0.2))

    #Add a second lstm layer
    model.add(LSTM(40, return_sequences=False))

    #Add a dense layer
    model.add(Dense(output_dim=1))

    #tanh is popular as an activation function for recurrent neural networks
    model.add(Activation('tanh'))

    #Use adam optimizer
    model.compile(loss='mse', optimizer='Adam')

    #fit the model
    model.fit(X_train, y_train, batch_size=10, epochs=5, validation_split=0.1)

    #predict
    predictions = model.predict(X_test)
    RMSE =  np.sqrt(mean_squared_error(y_test, predictions))

    return predictions, RMSE
    
def turn_to_dict(date, prediction):
    # res_dict = {str(date) +","+ str(prediction)}
    
    res_dict = dict(zip(date, prediction))
    return res_dict
    # result = json.dumps(result_set)
    

def analyzer(data):
    dates, prediction = getResults(data)
    
    #make them strings
    dates = [str(i) for i in dates]
    prediction = [str(i) for i in prediction]
    
    #make a list of dictionaries
    superlist = []
    for i in range(len(dates)):
        superlist.append({"Date": dates[i], "Prediction": prediction[i]})
        
    print(superlist)
    # formatted_result = turn_to_dict(dates,prediction)
    # print(formatted_result)
    
    # print("Got to end of analyzer")
    # print(json.dumps(formatted_result))
    return superlist
#----------------------------------------------------------------------------
#Call the function with the product you want to predict the next 10 days for.
# getResults("TIC TAC BIG PK FRUIT")
# analyzer("FRITO CHEETOS HOT")