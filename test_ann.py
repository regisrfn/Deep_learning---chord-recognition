import keras
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
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
dataset = pd.read_csv('./dataset_csv/chords_2.csv')
x_train = dataset.iloc[:, 0:12].values
y_train = dataset.iloc[:, 12].values
dataset = pd.read_csv('./dataset_csv/chords_test_2.csv')
x_test = dataset.iloc[:, 0:12].values
y_test = dataset.iloc[:, 12].values


# to create  confusion_matrix
Y_test = y_test 
# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# # Feature Scaling
# sc = MinMaxScaler(feature_range=(0, 1))
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

num_classes = y_train.shape[1]
print(f"Numero de classes = {num_classes}")
num_inputs = x_train.shape[1]

# build the model
model = classification_model(num_inputs)

# fit the model
model.fit(x_train, y_train, epochs=50, verbose=1)

# evaluate the model
# scores = model.evaluate(x_test, y_test, verbose=0)

# Predicting the Test set results
y_pred = model.predict(x_test)
y_pred = y_pred.argmax(axis=1)

# Making the Confusion Matrix
labels = range(10)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred, labels=labels)
acc = np.mean(y_pred == Y_test)
print(cm)
print(f"accuracy: {acc}")

model_name = 'model{:.4}'.format(acc*100)
# serialize model to JSON
model_json = model.to_json()
with open(f"./saved_models/{model_name}.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(f"./saved_models/{model_name}.h5")

print("Saved model to disk")
