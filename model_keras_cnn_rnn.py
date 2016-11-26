#!/usr/bin/python
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# load the dataset but only keep the top n words, zero the rest

def get_Feature_Label(hasJunk=True):
    data = np.genfromtxt('./input/featureMatrix.csv', delimiter=',')
    np.random.shuffle(data)

    X, y = data[:, :-1], data[:, -1]
    validation_ratio = 0.3
    # create train and test sets
    D = int(data.shape[0] * validation_ratio)  # total number of test data

    X_train, y_train = X[:-D], y[:-D]
    X_test, y_test = X[-D:], y[-D:]
    return X_train, y_train, X_test, y_test

def train(X_train, y_train, X_test, y_test):
    top_words = 5000
    embedLayer = 200
    # model = Sequential()
    # model.add(Embedding(top_words, embedLayer, input_length=40))
    # model.add(Convolution1D(nb_filter=embedLayer, filter_length=2, border_mode='same', activation='relu'))
    # model.add(MaxPooling1D(pool_length=2))
    # model.add(Flatten())
    # model.add(Dense(10, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())

    # create the model
    model = Sequential()
    model.add(Embedding(top_words, embedLayer, input_length=40))
    model.add(Flatten())
    # model.add(Dense(40, input_dim=40, init='normal', activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())


    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=2, batch_size=128, verbose=1)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_Feature_Label()
    train(X_train, y_train, X_test, y_test)
