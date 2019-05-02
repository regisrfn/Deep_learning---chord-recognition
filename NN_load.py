import keras
import pandas as pd
import numpy as np
import pickle

from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


with open("nn.file", "rb") as f:
    nn = pickle.load(f)

# Importing the dataset
dataset = pd.read_csv('./dataset_csv/chords_test.csv')
x_test = dataset.iloc[:, 0:12].values
y_test = dataset.iloc[:, 12].values

# to create  confusion_matrix
Y_test = y_test 
# one hot encode outputs
y_test = to_categorical(y_test)

#Feature Scaling
sc = StandardScaler()
x_test = sc.fit_transform(x_test)

num_classes = y_test.shape[1]
print(f"Numero de classes = {num_classes}")
num_inputs = x_test.shape[1]

predicted_y = nn.predict(x_test)
predicted_y = predicted_y.argmax(axis=1)
print(f"accuracy: {np.mean(predicted_y == Y_test)}")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, predicted_y)
print(cm)




