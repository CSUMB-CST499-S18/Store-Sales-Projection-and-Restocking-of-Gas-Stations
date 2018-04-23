import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error
import warnings
from pandas import DataFrame
from pandas import concat
import tensorflow as tf


#top level function to return the results
def getResults(itemName):

    df = openPreprocess("./data/sales.csv")

    itemSales = getSales(df, itemName)

    #get the dates
    dates = itemSales.index.values

    #keep the values from the dataframe
    X = itemSales.values

    #turn the time series into supervised data
    data = makeSupervised(X)

    X = data.iloc[:,0:8]
    y = data.iloc[:,8]

    #Break into training and testing data
    X_train = X[:355]
    X_val = X[355:385]
    X_test = X[385:]

    y_train = y[:355]
    y_val = y[355:385]
    y_test = y[385:]
    predicted_dates = dates[385:]

    #train the neural network and return the results
    results, RMSE = neuralNet(X_train, y_train, X_val, y_val, X_test, y_test)

    print("The predictions of the next 10 days are: ")

    for i in range(10):
        print(predicted_dates[i], results[i])

    #Compare with currently used method
    print("The neural net performed: ", simpleAverageRMSE(RMSE, X_test, y_test), "% times better compared to the previous method!")


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

    print("Shape after accounting for dates with no sales", item.shape)

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
    predicts_avg = [average]*10

    # print("Simple average prediction RMSE: ", np.sqrt(mean_squared_error(predicts_avg,y_test)))
    return np.sqrt(mean_squared_error(predicts_avg,y_test))*100/RMSE

#Build and run the Neural Net
def neuralNet(X_train, y_train, X_val, y_val, X_test, y_test):

    #define number of features, and nodes in hidden layers
    n_inputs = 8  # MNIST
    n_hidden1 = 100
    n_hidden2 = 100
    n_hidden3 = 50
    n_hidden4 = 10
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
        hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3",
                                  activation=tf.nn.elu)
        hidden4 = tf.layers.dense(hidden3, n_hidden4, name="hidden4",
                                  activation=tf.nn.elu)
        results = tf.layers.dense(hidden4, n_outputs, name="outputs")

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

    n_epochs = 2000 #times the data will be trained on

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):

            #Train on the training data
            sess.run(training_op, feed_dict={X: X_train, y: y_train})

            #Test on the validation data every 5000 epochs
            if epoch % 1000 == 0:
                print("Epoch", epoch, "RMSE (lower is better) =", np.sqrt(mse.eval(feed_dict={X: X_val, y: y_val})))

        #Finally, test on the test dataset
        predictions = results.eval(feed_dict={X: X_test, y: y_test})
        testRMSE = np.sqrt(mse.eval(feed_dict={X: X_test, y: y_test}))

        print("Test RMSE: ", testRMSE)
        return predictions, testRMSE

#----------------------------------------------------------------------------
#Call the function with the product you want to predict the next 10 days for.
getResults("MARL GOLD")
