import keras
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# import the data
from keras.datasets import mnist

# Importing the dataset
dataset = pd.read_csv('./chords.csv')
X = dataset.iloc[:, 0:12].values
Y = dataset.iloc[:, 12].values

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)

# to create  confusion_matrix
Y_test = y_test 
# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# # Feature Scaling
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

num_classes = y_test.shape[1]
print(f"Numero de classes = {num_classes}")
num_inputs = x_test.shape[1]

# define classification model
def classification_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='sigmoid', input_shape=(num_inputs,)))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(num_classes, activation='sigmoid'))
    
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# build the model
model = classification_model()

# fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, verbose=2)

# evaluate the model
scores = model.evaluate(x_test, y_test, verbose=0)

# Predicting the Test set results
y_pred = model.predict(x_test)
y_pred = y_pred.argmax(axis=1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)