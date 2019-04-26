import keras
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# define classification model
def classification_model(input_size):
    # create model
    model = Sequential()
    model.add(Dense(50, activation='sigmoid', input_shape=(input_size,)))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))

    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



# Importing the dataset
dataset = pd.read_csv('./chords.csv')
x_train = dataset.iloc[:, 0:12].values
y_train = dataset.iloc[:, 12].values
dataset = pd.read_csv('./chords_test.csv')
x_test = dataset.iloc[:, 0:12].values
y_test = dataset.iloc[:, 12].values


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

# build the model
model = classification_model(num_inputs)

# fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=35, verbose=2)

# evaluate the model
scores = model.evaluate(x_test, y_test, verbose=0)

# Predicting the Test set results
y_pred = model.predict(x_test)
y_pred = y_pred.argmax(axis=1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)
# print(f"accuracy: {np.mean(y_pred == Y_test)}")


# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")