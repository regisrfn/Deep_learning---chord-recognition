import keras
import pandas as pd
import numpy as np
import pickle
import neural_network.NeuralNework as classification

from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# Importing the dataset
dataset = pd.read_csv('./dataset_csv/chords.csv')
x_train = dataset.iloc[:, 0:12].values
y_train = dataset.iloc[:, 12].values
dataset = pd.read_csv('./dataset_csv/chords_test.csv')
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

num_classes = y_train.shape[1]
print(f"Numero de classes = {num_classes}")
num_inputs = x_train.shape[1]

nn = classification.NeuralNetwork(x_train,y_train)
nn.add_layer(size=50,input_size=num_inputs)
nn.add_layer(size=50)
nn.add_layer(size=num_classes)


epochs = 100000
for epoch in range(epochs):
    nn.feedforward()
    nn.backprop(learning_rate=0.5)
    print(f"epoch {epoch+1}/{epochs}")
    predicted_y = nn.predict(x_test)
    predicted_y = predicted_y.argmax(axis=1)
    print(f"accuracy: {np.mean(predicted_y == Y_test)}")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, predicted_y)
print(cm)

# save network
import pickle 
with open("nn.file", "wb") as f:
    pickle.dump(nn, f, pickle.HIGHEST_PROTOCOL)
