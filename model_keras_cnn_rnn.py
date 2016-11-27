#!/usr/bin/python
import numpy as np
import random
import operator
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix

# load the dataset but only keep the top n words, zero the rest

def value2Categorical(y, clusters=2):
    label = np.copy(y)
    label[y<np.percentile(y, 100/clusters)] = 0
    for i in range(1, clusters):
        label[y>np.percentile(y, 100*i/clusters)] = i
    return label



def get_Feature_Label(clusters=2, hasJunk=True):
    data = np.genfromtxt('./input/featureMatrix.csv')
    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1]
    label = to_categorical(value2Categorical(y, clusters)).astype("int")

    validation_ratio = 0.5
    D = int(data.shape[0] * validation_ratio)  # total number of test data
    X_train, y_train, X_test, y_test = X[:-D], label[:-D,:], X[-D:], label[-D:,:]
    return X_train, y_train, X_test, y_test



def embeddingNN(X_train, y_train, X_test, y_test, clusters=2, embedLayer=200, middle = 100):
    top_words = 2001
    lossType = 'binary_crossentropy' if y_test.shape[1] == 2 else 'categorical_crossentropy'
    model = Sequential()
    model.add(Embedding(top_words, embedLayer, input_length=X_train.shape[1]))
    model.add(Flatten())
    model.add(Dense(middle, activation='relu'))
    model.add(Dense(clusters, activation='sigmoid'))
    model.compile(loss=lossType, optimizer='adam', metrics=['accuracy'])
    return model

def embeddingCNN(X_train, y_train, X_test, y_test, clusters=2, embedLayer=200, middle = 100):
    top_words = 2001
    lossType = 'binary_crossentropy' if y_test.shape[1] == 2 else 'categorical_crossentropy'
    model = Sequential()
    model.add(Embedding(top_words, embedLayer, input_length=X_train.shape[1]))
    model.add(Convolution1D(nb_filter=embedLayer, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(middle, activation='relu'))
    model.add(Dense(clusters, activation='sigmoid'))
    model.compile(loss=lossType, optimizer='adam', metrics=['accuracy'])
    return model


def model_selection(): # random sampling is better than grid search
    sampling_Num = 1
    clusters = 2
    X_train, y_train, X_test, y_test = get_Feature_Label(clusters=clusters)
    
    model_list = []
    for embeddings in range(30, 400, 20):
        for middle in range(50, 400, 30):
            model_list.append((embeddings, middle))

    rand_smpl = [model_list[i] for i in sorted(random.sample(xrange(len(model_list)), sampling_Num))]
    
    performance = {}
    cnt = {}
    scores = {}
    for pars in rand_smpl:
        model = embeddingNN(X_train, y_train, X_test, y_test, clusters, embeddings, middle)
        print(model.summary())
        model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=2, batch_size=128, verbose=1)
        # Final evaluation of the model
        score = model.evaluate(X_test, y_test, verbose=0)
        # calculate predictions
        predictions = np.argmax(model.predict(X_test), axis=-1)
        print(confusion_matrix(np.argmax(y_test, axis=-1), predictions))
        print("Accuracy: %.2f%%" % (score[1]*100))
        performance[pars] = performance.get(pars, 0) + score[1]*100
        cnt[pars] = cnt.get(pars, 0) + 1

    for pars in cnt:
        scores[pars] = round(performance[pars] / cnt[pars], 2)
    scores = sorted(scores.items(), key=operator.itemgetter(1))
    for num, i in enumerate(scores):
        print("pars=%s, score=%.2f, trials=%d" % (i[0], i[1], cnt[i[0]]))
        if num > 10: break

def main():
    model_selection()
    


if __name__ == "__main__":
    main()
