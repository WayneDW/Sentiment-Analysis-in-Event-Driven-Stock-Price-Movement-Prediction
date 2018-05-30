#! /usr/bin/env python3
import os

import argparse
import datetime
import torch
import model
import train
import numpy as np

import util

parser = argparse.ArgumentParser(description='CNN-based Financial News Classifier')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=2, help='number of epochs for train [default: 2]')
parser.add_argument('-batch-size', type=int, default=256, help='batch size for training [default: 256]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=256, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()

# load tokenized features
data = np.genfromtxt('./input/featureMatrix_train')
test = np.genfromtxt('./input/featureMatrix_test')
np.random.shuffle(data)
X, y = data[:, :-1], data[:, -1]
label = util.value2int_simple(y).astype("int") # using direction to label
#label = to_categorical(value2int(y, clusters)).astype("int") # using quantile to label
validation_ratio = 0.2
X = X.astype('float32')
D = int(data.shape[0] * validation_ratio)  # total number of validation data
X_train, y_train, X_valid, y_valid = X[:-D], label[:-D], X[-D:], label[-D:]
X_test, y_test = test[:, :-1], test[:, -1]

print("Positive News Ratio", sum(y_test > 0) * 1. / (sum(y_test > 0) + sum(y_test < 0)))
X_test = X_test.astype('float32')
y_test = util.value2int_simple(y_test).astype("int")


# update args and print
args.embed_num = 2001
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
    cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()
        

# train or predict
if args.predict is not None:
    label = train.predict(args.predict, cnn, text_field, label_field, args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        train.eval(test_iter, cnn, args) 
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train.train(X_train, y_train, X_valid, y_valid, cnn, args)
        train.eval(X_test, y_test, cnn, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')

