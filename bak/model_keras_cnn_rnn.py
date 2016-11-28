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


def value2Categorical(y, clusters=2):
    label = np.copy(y)
    label[y<np.percentile(y, 100/clusters)] = 0
    for i in range(1, clusters):
        label[y>np.percentile(y, 100*i/clusters)] = i
    return label



def get_Feature_Label(clusters=2, hasJunk=True):
    data = np.genfromtxt('./input/featureMatrix_body.csv')
    test = np.genfromtxt('./input/featureMatrixTest_body.csv')
    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1]
    label = to_categorical(value2Categorical(y, clusters)).astype("int")

    validation_ratio = 0.1
    D = int(data.shape[0] * validation_ratio)  # total number of test data
    X_train, y_train, X_valid, y_valid = X[:-D], label[:-D,:], X[-D:], label[-D:,:]

    
    X_test, y_test = test[:, :-1], test[:, -1]
    y_test = to_categorical(value2Categorical(y_test, clusters)).astype("int")

    return X_train, y_train, X_valid, y_valid, X_test, y_test



def embeddingNN(shape, clusters=2, embedLayer=200, middle = 100):
    top_words = 2001
    lossType = 'binary_crossentropy' if clusters == 2 else 'categorical_crossentropy'
    model = Sequential()
    model.add(Embedding(top_words, embedLayer, input_length=shape))
    model.add(Flatten())
    model.add(Dense(middle, activation='relu'))
    model.add(Dense(clusters, activation='sigmoid'))
    model.compile(loss=lossType, optimizer='adam', metrics=['accuracy'])
    return model

def embeddingCNN(shape, clusters=2, embedLayer=200, middle = 100):
    top_words = 2001
    lossType = 'binary_crossentropy' if clusters == 2 else 'categorical_crossentropy'
    model = Sequential()
    model.add(Embedding(top_words, embedLayer, input_length=shape))
    model.add(Convolution1D(nb_filter=embedLayer, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(middle, activation='relu'))
    model.add(Dense(clusters, activation='sigmoid'))
    model.compile(loss=lossType, optimizer='adam', metrics=['accuracy'])
    return model


def model_selection(): # random sampling is better than grid search
    sampling_Num = 5
    clusters = 2
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_Feature_Label(clusters=clusters)
    model_list = []
    for embeddings in range(20, 200, 10):
        for middle in range(100, 300, 5):
            model_list.append((embeddings, middle))

    # rand_smpl = [model_list[i] for i in sorted(random.sample(xrange(len(model_list)), sampling_Num))]
    rand_smpl = [model_list[random.randint(0,len(model_list) - 1)] for i in range(sampling_Num)]
    
    performance = {}
    cnt = {}
    scores = {}
    for pars in rand_smpl:
        embeddings, middle = pars
        model = embeddingNN(X_train.shape[1], clusters, embeddings, middle)
        print(model.summary())
        model.fit(X_train, y_train, validation_data=(X_valid, y_valid), nb_epoch=2, batch_size=128, verbose=2)
        # Final evaluation of the model
        score = model.evaluate(X_test, y_test, verbose=0)
        # calculate predictions
        predictions = np.argmax(model.predict(X_test), axis=-1)
        conf = confusion_matrix(np.argmax(y_test, axis=-1), predictions)
        print("Test on %d samples" % (len(y_test)))
        print(conf)
        conf = np.array(conf)
        for i in range(clusters):
            print("Label %d Precision: %.2f%%" % (i, conf[i,i] * 100.0 / sum(conf[:,i])))
        performance[pars] = performance.get(pars, 0) + conf[i,i] * 100.0 / sum(conf[:,i])
        cnt[pars] = cnt.get(pars, 0) + 1

    for pars in cnt:
        scores[pars] = round(performance[pars] / cnt[pars], 2)
    sorted_Scores = sorted(scores.items(), key=operator.itemgetter(1))
    for num, i in enumerate(sorted_Scores):
        print("pars=%s, score, %.2f, trials=%d" % (i[0], i[1], cnt[i[0]]))

def main():
    model_selection()
    


if __name__ == "__main__":
    main()
