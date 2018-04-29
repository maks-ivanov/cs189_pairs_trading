from __future__ import print_function
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
# fix random seed for reproducibility
np.random.seed(7)

X = np.load('X.npy')
Y = np.load('Y.npy')

X_train, X_test, y_train, y_test = train_test_split(X, Y)
# create the model
model = Sequential()
model.add(LSTM(264, input_shape = (100, 1)))
model.add(Dense(10, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, nb_epoch=10, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
