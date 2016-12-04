#!/usr/bin/python
import numpy as np
import random
import operator
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix


def value2int(y, clusters=2):
    label = np.copy(y)
    label[y<np.percentile(y, 100/clusters)] = 0
    for i in range(1, clusters):
        label[y>np.percentile(y, 100*i/clusters)] = i
    return label

def get_Feature_Label(clusters=2, hasJunk=True):
    data = np.genfromtxt('./input/featureMatrix_train.csv')
    test = np.genfromtxt('./input/featureMatrix_test.csv')
    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1]
    label = to_categorical(value2int(y, clusters)).astype("int")

    validation_ratio = 0.2
    X = X.reshape(X.shape[0], 100, 30, 1).astype('float32')
    D = int(data.shape[0] * validation_ratio)  # total number of validation data
    X_train, y_train, X_valid, y_valid = X[:-D], label[:-D,:], X[-D:], label[-D:,:]

    
    X_test, y_test = test[:, :-1], test[:, -1]
    X_test = X_test.reshape(X_test.shape[0], 100, 30, 1).astype('float32')
    y_test = to_categorical(value2int(y_test, clusters)).astype("int")

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def CNN(clusters):
    model = Sequential()
    model.add(Convolution2D(128, 100, 3, border_mode='valid', input_shape=(100, 30, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 28)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(clusters, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def evaluate(model, clusters, X_train, y_train, X_valid, y_valid, X_test, y_test):
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), nb_epoch=2, batch_size=128, verbose=2)
    # Final evaluation of the model
    score = model.evaluate(X_test, y_test, verbose=0)
    

    predictions = np.argmax(model.predict(X_valid), axis=-1)
    conf = confusion_matrix(np.argmax(y_valid, axis=-1), predictions)
    print(conf)
    for i in range(clusters):
        print("Valid Label %d Precision, %.2f%%" % (i, conf[i,i] * 100.0 / sum(conf[:,i])))

    # calculate predictions
    predictions = np.argmax(model.predict(X_test), axis=-1)
    conf = confusion_matrix(np.argmax(y_test, axis=-1), predictions)
    print("Test on %d samples" % (len(y_test)))
    print(conf)
    for i in range(clusters):
        print("Test Label %d Precision, %.2f%%" % (i, conf[i,i] * 100.0 / sum(conf[:,i])))


def model_selection(): # random sampling is better than grid search
    sampling_Num = 5
    clusters = 2
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_Feature_Label(clusters=clusters)

    model = CNN(clusters)
    print(model.summary())
    for i in range(100):
        evaluate(model, clusters, X_train, y_train, X_valid, y_valid, X_test, y_test)
        

def main():
    model_selection()
    


if __name__ == "__main__":
    main()
