import keras
import pandas as pd
import numpy as np
import pickle
import neural_network.NeuralNework as classification

from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('./dataset_csv/chords.csv')
X = dataset.iloc[:, 0:12].values
Y = dataset.iloc[:, 12].values

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.1, random_state=0)


# to create  confusion_matrix
Y_test = y_test 
# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

num_classes = y_train.shape[1]
print(f"Numero de classes = {num_classes}")
num_inputs = x_train.shape[1]

nn = classification.NeuralNetwork(x_train,y_train)
nn.add_layer(size=100,input_size=num_inputs)
nn.add_layer(size=50)
nn.add_layer(size=num_classes)


epochs = 10000
for epoch in range(epochs):
    step = epoch + 1
    lr = np.power(0.5, np.ceil(np.log10(step)))
    nn.feedforward()
    nn.backprop(learning_rate=lr)
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
