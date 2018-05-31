#! /usr/bin/env python3
import os

import argparse
import datetime
import torch
import model
import train
import numpy as np

import util
import json
from SGHMC_Bayesian import sghmc

parser = argparse.ArgumentParser(description='CNN-based Financial News Classifier')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=50, help='number of epochs for train [default: 50]')
parser.add_argument('-batch-size', type=int, default=128, help='batch size for training [default: 128]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=64, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
#parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
parser.add_argument('-static', action='store_true', default=True, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-vocabs', type=int, default=1000, help='total number of vocabularies [default: 1000]')
parser.add_argument('-words', type=int, default=20, help='max number of words in a sentence [default: 20]')
parser.add_argument('-date', type=str, default="20180525", help='date to be tested')


args = parser.parse_args()

# load tokenized features
data = np.genfromtxt('./input/featureMatrix_train')
test = np.genfromtxt('./input/featureMatrix_test')
np.random.shuffle(data)
X, y = data[:, :-1], data[:, -1]
label = util.value2int_simple(y).astype("int") # using direction to label
#label = to_categorical(value2int(y, clusters)).astype("int") # using quantile to label
validation_ratio = 0.05
X = X.astype('float32')
D = int(data.shape[0] * validation_ratio)  # total number of validation data
X_train, y_train, X_valid, y_valid = X[:-D], label[:-D], X[-D:], label[-D:]
X_test, y_test = test[:, :-1], test[:, -1]

print("Positive News Ratio", sum(y_test > 0) * 1. / (sum(y_test > 0) + sum(y_test < 0)))
X_test = X_test.astype('float32')
y_test = util.value2int_simple(y_test).astype("int")


# update args and print
args.class_num = 2
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


# model
cnn = model.CNN_Text(args)
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load("./snapshot/" + args.snapshot))


if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()

# train or predict
if args.predict is not None:
    with open('./input/word2idx', 'r') as file:
        word2idx = json.load(file)

    stopWords = set()
    with open('./input/stopWords') as file:
        for word in file:
            stopWords.add(word.strip())

    with open('./news/2018/news_' + args.date + '.csv') as f:
        for num, line in enumerate(f):
            line = line.strip().split(',')
            if len(line) != 6:
                continue
            ticker, name, day, headline, body, newsType = line
            if newsType != 'topStory': # newsType: [topStory, normal]
                continue # skip normal news
            tokens = util.tokenize_news(headline, stopWords)
            tokens = [word2idx[t] if t in word2idx else word2idx['UNKNOWN'] for t in tokens]
            if len(tokens) < 5 or tokens == [word2idx['UNKNOWN']] * len(tokens): # tokens cannot be too short or unknown
                continue
            feature = torch.LongTensor([tokens])
            predictor = train.predict(cnn, feature, args.words, word2idx, stopWords, args.cuda)
            print(name, headline, predictor)
elif args.test:
    try:
        train.eval(test_iter, cnn, args) 
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train.train(X_train, y_train, X_valid, y_valid, X_test, y_test, cnn, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')

