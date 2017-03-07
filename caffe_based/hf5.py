from __future__ import print_function
import numpy as np
import h5py
import os
import sys

import pandas as pd

DIR = "/scratch/radon/d/deng106/CNNStatisticalModel/distributions/"

num_distributions = sys.argv[1]
num_sampleSize = sys.argv[2]

dataname = DIR + "data/feature_" + num_distributions + "_distributions_" + num_sampleSize
targetname = DIR + "data/label_" + num_distributions + "_distributions_" + num_sampleSize
OUTPUT = DIR + "input/" + num_distributions + "_distributions_" + num_sampleSize + "_sample_size_"

data = np.genfromtxt(dataname, delimiter=',')
size = data.shape[0]

sample_dimension = int(np.sqrt(int(num_sampleSize)))
data = np.reshape(data, (size, 1, sample_dimension,  sample_dimension))


import random
a = range(0, size)
random.shuffle(a)

test_size = int(round(size*0.8))

data_train = data[a[:test_size]]
data_test = data[a[test_size:]]


labels = pd.read_table(targetname, delimiter=',', header=None)


parameters = np.array(labels.iloc[:, 0])
distributions = np.array(labels.iloc[:, 1], dtype=np.int8)
#distributions = np.reshape(distributions, (size, 1, 1, 1))



parameters_train = parameters[a[:test_size]]
parameters_test = parameters[a[test_size:]]
distributions_train = distributions[a[:test_size]]
distributions_test = distributions[a[test_size:]]


# prepare parameter data

with h5py.File(OUTPUT + 'train_parameters.h5', 'w') as f:
    f['data'] = data_train
    f['label'] = parameters_train


with h5py.File(OUTPUT + 'test_parameters.h5', 'w') as f:
    f['data'] = data_test
    f['label'] = parameters_test

with open(OUTPUT + 'train-parameters-path.txt', 'w') as f:
    print(OUTPUT + 'train_parameters.h5', file = f)

with open(OUTPUT + 'test-parameters-path.txt', 'w') as f:
     print(OUTPUT + 'test_parameters.h5', file = f)


# prepare distribution data

with h5py.File(OUTPUT + 'train_distributions.h5', 'w') as f:
    f['data'] = data_train
    f['label'] = distributions_train

with h5py.File(OUTPUT + 'test_distributions.h5', 'w') as f:
    f['data'] = data_test
    f['label'] = distributions_test

with open(OUTPUT + 'train-distributions-path.txt', 'w') as f:
    print(OUTPUT + 'train_distributions.h5', file = f)

with open(OUTPUT + 'test-distributions-path.txt', 'w') as f:
     print(OUTPUT + 'test_distributions.h5', file = f)


