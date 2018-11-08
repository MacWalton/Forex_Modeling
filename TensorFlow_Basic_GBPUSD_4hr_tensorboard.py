# -*- coding: utf-8 -*-
"""
Forex Data Regression Analysis on GBPUSD 4hr data
Created on Sat Oct 13 13:04:06 2018

@author: Macalister Walton

Note: to run Tensorboard - cmd to logs directory and
type" tensorboard --logdir=logs/

creates a local server: TensorBoard 0.4.0rc3 at http://DESKTOP-ESLLRD2:6006 (Press CTRL+C to quit)
use http://DESKTOP-ESLLRD2:6006  to tensorbord
"""
import tensorflow as tf
from keras.callbacks import TensorBoard
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
import time


EPOCHS = 300
LearningRate = 0.06

NAME = "GBPUSD_Close_Model_4hr{}".format(int(time.time())) # create Names for models

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME)) # writes callback to file


# uses the a fraction of the GPU for processing the model eg. 33.3%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Data import and create test and training data sets
df=pd.read_csv('GBPUSD_4hr.csv')
X = df.iloc[:,2:6].values
X=pd.DataFrame(X)
y = df.iloc[:,6].values
y = pd.DataFrame(y)

df.drop(X.index[-1]) # remove the last record with no y label
df.drop(y.index[1]) # adjust the close to 4hrs in the future

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

print("Training set: {}".format(X_train.shape))  # 17325 examples, 4 features
print("Test set:  {}".format(X_test.shape))   # 4332 samples, 4 features


# #Normalise the data
mean = X_train.mean(axis=0)
std = X_train.std(axis = 0)
X_train= (X_train -mean)/std
X_test = (X_test-mean)/std



# #Build a model on the size and type of Network required
def build_model():
    model = keras.Sequential([
            keras.layers.Dense(16, activation = tf.nn.sigmoid,
                                input_shape=(X_train.shape[1],)),
            keras.layers.Dense(64, activation=tf.nn.sigmoid),
            keras.layers.Dense(32, activation=tf.nn.sigmoid),
            keras.layers.Dense(1)
            ])

    optimizer = tf.train.RMSPropOptimizer(LearningRate)

    model.compile(loss ='mse',
                  optimizer = optimizer,
                  metrics=['mae'])
    return model

model = build_model()
model.summary()

#Train and Store training Stats
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
#The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)

history = model.fit(X_train, y_train, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop,PrintDot(),tensorboard])


#Let's see how did the model performs on the test set
[loss, mae] =model.evaluate(X_test,y_test,verbose=0)
print("Testing set Mean Abs Error: {:7.5f}pips".format(mae))

test_predictions = model.predict(X_test).flatten()

plt.scatter(y_test, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())

_ = plt.plot([-300,300],[-300,300])






