# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('./dataset_csv/chords.csv')
test = pd.read_csv('./dataset_csv/chords_test_violin.csv')
X = dataset.iloc[:, 0:12].values
y = dataset.iloc[:, 12].values

X_test = test.iloc[:, 0:12].values
y_test = test.iloc[:, 12].values

# #Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X = sc.fit_transform(X)
#X_test = sc.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=20, metric='minkowski', p=2)
classifier.fit(X, y)

# Predicting the Test set result
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


print(np.mean(y_pred == y_test))
print(cm)

