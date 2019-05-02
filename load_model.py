import keras
import pandas as pd
import numpy as np

from keras.models import model_from_json
from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# define classification model
def classification_model():

    # create model
    # load json and create model
    json_file = open('./saved_models/model99.33.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("./saved_models/model99.33.h5")
    print("Loaded model from disk")

    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model


# Importing the dataset
dataset = pd.read_csv('./dataset_csv/chords.csv')
x_train = dataset.iloc[:, 0:12].values
y_train = dataset.iloc[:, 12].values
dataset = pd.read_csv('./dataset_csv/chords_test_piano.csv')
x_test = dataset.iloc[:, 0:12].values
y_test = dataset.iloc[:, 12].values


# to create  confusion_matrix
Y_test = y_test
# one hot encode outputs
y_test = to_categorical(y_test)

# Feature Scaling
# sc = MinMaxScaler(feature_range=(-1, 1))
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

num_classes = y_test.shape[1]
print(f"Numero de classes = {num_classes}")

# build the model
model = classification_model()
# Predicting the Test set results
y_pred = model.predict(x_test)
y_pred = y_pred.argmax(axis=1)

# Making the Confusion Matrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)
print(f"accuracy: {np.mean(y_pred == Y_test)}")

