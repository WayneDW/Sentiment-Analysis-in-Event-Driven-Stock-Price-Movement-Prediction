#!/usr/bin/env python3
import os
import sys
import copy
import re
import time
import datetime

from urllib.request import urlopen

import numpy as np

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import json
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

# training with SGLD with annealing and save models
def train(X_train, y_train, X_valid, y_valid, X_test, y_test, model, args):
    model.train()
    batch = args.batch_size

    parameters = [parameter for parameter in model.parameters()]


    set_scale = [parameter.data.std().item() for parameter in model.parameters()]
    set_scale = [scale / max(set_scale) for scale in set_scale] # normalize
    for epoch in range(1, args.epochs+1):
        corrects = 0
        epsilon = args.lr * ((epoch * 1.0) ** (-0.333)) # optimal decay rate
        for idx in range(int(X_train.shape[0]/batch) + 1):
            feature = torch.LongTensor(X_train[(idx*batch):(idx*batch+batch),])
            target = torch.LongTensor(y_train[(idx*batch):(idx*batch+batch)])
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            model.zero_grad()
            loss.backward()

            for layer_no, param in enumerate(model.parameters()):
                if args.static and layer_no == 0: # fixed embedding layer cannot update
                    continue
                # by default I assume you train the models using GPU
                noise = torch.cuda.FloatTensor(param.data.size()).normal_() * np.sqrt(epsilon / args.t)
                #noise = torch.cuda.FloatTensor(param.data.size()).normal_() * set_scale[layer_no]
                parameters[layer_no].data += (- epsilon / 2 * param.grad + noise)

            corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum().item()
            accuracy = 100.0 * corrects / batch / (idx + 1)
            sys.stdout.write('\rEpoch[{}] Batch[{}] - loss: {:.4f}  acc: {:.2f}%({}/{}) tempreture: {}'.format(
                             epoch, idx, loss.item(), accuracy, corrects, batch * (idx + 1), int(args.t)))
            args.t = args.t + 1 # annealing
        if epoch % 5 != 0:
            continue
        '''
        try:
            set_scale = [parameter.grad.data.std().item() for parameter in model.parameters()]
            set_scale = [scale / max(set_scale) for scale in set_scale] # normalize
        except:
            set_scale = [parameter.data.std().item() for parameter in model.parameters()]
            set_scale = [scale / max(set_scale) for scale in set_scale] # normalize
        '''
        save(model, args.save_dir, epoch)
        print()
        eval(X_valid, y_valid, model, 'Validation', args)
        eval(X_test, y_test, model, 'Testing   ', args)


def eval(X, y, model, term, args):
    model.eval()
    corrects, TP, avg_loss = 0, 0, 0
    correct_part, total_part = {0.2:0, 0.4:0}, {0.2:1e-16, 0.4:1e-16}
    batch = args.batch_size

    for idx in range(int(X.shape[0]/batch) + 1):
        feature = torch.LongTensor(X[(idx*batch):(idx*batch+batch),])
        target = torch.LongTensor(y[(idx*batch):(idx*batch+batch)])
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)
        avg_loss += loss.data.item()
        predictor = torch.exp(logit[:, 1]) / (torch.exp(logit[:, 0]) + torch.exp(logit[:, 1]))
        for xnum in range(1, 3):
            thres = round(0.2 * xnum, 1)
            idx_thres = (predictor > 0.5 + thres) + (predictor < 0.5 - thres)
            correct_part[thres] += (torch.max(logit, 1)[1][idx_thres] == target.data[idx_thres]).sum().item()
            total_part[thres] += idx_thres.sum().item()

        corrects += (torch.max(logit, 1)[1] == target.data).sum().item()
        TP += (((torch.max(logit, 1)[1] == target.data).int() + (torch.max(logit, 1)[1]).int()) == 2).sum().item()

    size = y.shape[0]
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    # TP, TN: True Positive/True Negative
    print('         {} - loss: {:.4f} acc: {:.2f}%({}/{}) {:.2f}%({}/{}) {:.2f}%({}/{}) TP/TN: ({}/{}) \n'.format(term,
          avg_loss, accuracy, corrects, size, 100.0 * correct_part[0.2] / total_part[0.2], correct_part[0.2], int(total_part[0.2]), 
          100.0 * correct_part[0.4] / total_part[0.4], correct_part[0.4], int(total_part[0.4]), TP, corrects - TP))
    return accuracy

def bma_eval(X, y, mymodels, term, args):
    
    corrects, TP, avg_loss = 0, 0, 0
    correct_part, total_part = {0.2:0, 0.4:0}, {0.2:1e-16,0.4:1e-16}
    batch = args.batch_size

    for model in mymodels:
        model.eval()
        for idx in range(int(X.shape[0]/batch) + 1):
            feature = torch.LongTensor(X[(idx*batch):(idx*batch+batch),])
            target = torch.LongTensor(y[(idx*batch):(idx*batch+batch)])
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            logit = model(feature)
            loss = F.cross_entropy(logit, target, size_average=False)
            avg_loss += loss.data.item() / (len(mymodels) * 1.0)
            predictor = torch.exp(logit[:, 1]) / (torch.exp(logit[:, 0]) + torch.exp(logit[:, 1]))
            for xnum in range(1, 3):
                thres = round(0.2 * xnum, 1)
                idx_thres = (predictor > 0.5 + thres) + (predictor < 0.5 - thres)
                correct_part[thres] += (torch.max(logit, 1)[1][idx_thres] == target.data[idx_thres]).sum().item() / (len(mymodels) * 1.0)
                total_part[thres] += idx_thres.sum().item() / (len(mymodels) * 1.0)
            corrects += (torch.max(logit, 1)[1] == target.data).sum().item() / (len(mymodels) * 1.0)
            TP += (((torch.max(logit, 1)[1] == target.data).int() + (torch.max(logit, 1)[1]).int()) == 2).sum().item()

    size = y.shape[0]
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    TP = TP * 1.0 / (len(mymodels) * 1.0)
    print('         {} - loss: {:.4f} acc: {:.2f}%({}/{}) {:.2f}%({}/{}) {:.2f}%({}/{}) TP/TN: ({}/{}) \n'.format(term,
            avg_loss, accuracy, corrects, size, 100.0 * correct_part[0.2] / total_part[0.2], correct_part[0.2], int(total_part[0.2]), 
            100.0 * correct_part[0.4] / total_part[0.4], correct_part[0.4], int(total_part[0.4]), TP, corrects - TP))
    return accuracy

def predictor_preprocess(cnn, args):
    # load trained thinning samples (Bayesian CNN models) from input/models/
    mymodels = []
    for num, each_model in enumerate(os.listdir(args.save_dir)):
        print(args.save_dir + each_model)
        if args.cuda:
            cnn.load_state_dict(torch.load(args.save_dir + each_model))
        else:
            cnn.load_state_dict(torch.load(args.save_dir + each_model, map_location=lambda storage, loc: storage))
        mymodels.append(copy.deepcopy(cnn))
        if num > 30: # in case memory overloads
            break

    with open('./input/word2idx', 'r') as file:
        word2idx = json.load(file)

    stopWords = set()
    with open('./input/stopWords') as file:
        for word in file:
            stopWords.add(word.strip())
    return(mymodels, word2idx, stopWords)

def predict(sentence, mymodels, word2idx, stopWords, args):
    tokens = tokenize_news(sentence, stopWords)
    tokens = [word2idx[t] if t in word2idx else word2idx['UNKNOWN'] for t in tokens]
    if len(tokens) < 5 or tokens == [word2idx['UNKNOWN']] * len(tokens): # tokens cannot be too short or unknown
        signal = 'Unknown'
    else:
        feature = torch.LongTensor([tokens])
        logits = []
        for model in mymodels:
            model.eval()
            if args.cuda:
                feature = feature.cuda()
            logit = model(feature)
            predictor = torch.exp(logit[:, 1]) / (torch.exp(logit[:, 0]) + torch.exp(logit[:, 1]))
            logits.append(predictor.item())
        signal = signals(np.mean(logits))
    return(signal)


def daily_predict(cnn, args):
    mymodels, word2idx, stopWords = predictor_preprocess(cnn, args)
    output = './input/news/' + args.date[:4] + '/news_' + args.date + '.csv'
    fout = open(output + '_bak', 'w')
    with open(output) as f:
        for num, line in enumerate(f):
            line = line.strip().split(',')
            if len(line) == 6:
                ticker, name, day, headline, body, newsType = line
            elif len(line) == 7:
                ticker, name, day, headline, body, newsType, signal = line
            else:
                continue

            #if newsType != 'topStory': # newsType: [topStory, normal]
            #    signal = 'Unknown'
            content = headline + ' ' + body
            signal = predict(content, mymodels, word2idx, stopWords, args)
            fout.write(','.join([ticker, name, day, headline, body, newsType, signal]) + '\n')
    fout.close()
    print('change file name')
    print('mv ' + output + '_bak ' + output)
    os.system('mv ' + output + '_bak ' + output)


def save(model, save_dir, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = '{}/model_{}.pt'.format(save_dir,steps)
    torch.save(model.state_dict(), save_path)

def signals(digit):
    strong_signal = 0.4
    unknown_thres = 0.05
    if digit > 0.5 + strong_signal:
        return('Strong Buy')
    elif digit > 0.5 + unknown_thres:
        return('Buy')
    elif digit > 0.5 - unknown_thres:
        return('Unknown')
    elif digit > 0.5 - strong_signal:
        return('Sell')
    else:
        return('Strong Sell')

def padding(sentencesVec, keepNum):
    shape = sentencesVec.shape[0]
    ownLen = sentencesVec.shape[1]
    if ownLen < keepNum:
        return np.hstack((np.ones([shape, keepNum-ownLen]), sentencesVec)).flatten()
    else:
        return sentencesVec[:, -keepNum:].flatten()


def dateGenerator(numdays): # generate N days until now, eg [20151231, 20151230]
    base = datetime.datetime.today()
    date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
    for i in range(len(date_list)):
        date_list[i] = date_list[i].strftime("%Y%m%d")
    return set(date_list)


def generate_past_n_days(numdays):
    """Generate N days until now, e.g., [20151231, 20151230]."""
    base = datetime.datetime.today()
    date_range = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
    return [x.strftime("%Y%m%d") for x in date_range]

def unify_word(word):  # went -> go, apples -> apple, BIG -> big
    """unify verb tense and noun singular"""
    ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
    for wt in [ADJ, ADJ_SAT, ADV, NOUN, VERB]:
        try:
            word = WordNetLemmatizer().lemmatize(word, pos=wt)
        except:
            pass
    return word.lower()

def digit_filter(word):
    check = re.match(r'\d*\.?\d*', word).group()
    if check == "":
        return word
    else:
        return ""

def unify_word_meaning(word):
    if word in ["bigger-than-expected", "higher-than-expected", "better-than-expected", "stronger-than-expected"]:
        return "better"
    elif word in ["smaller-than-expected", "lower-than-expected", "weaker-than-expected", "worse-than-expected"]:
        return "lower"
    elif word in ["no", "not", "n't"]:
        return "not"
    else:
        return word

def get_soup_with_repeat(url, repeat_times=3, verbose=True):
    for i in range(repeat_times): # repeat in case of http failure
        try:
            time.sleep(np.random.poisson(3))
            response = urlopen(url)
            data = response.read().decode('utf-8')
            return BeautifulSoup(data, "lxml")
        except Exception as e:
            if i == 0:
                print(e)
            if verbose:
                print('retry...')
            continue

def tokenize_news(headline, stopWords):
    tokens = nltk.word_tokenize(headline) #+ nltk.word_tokenize(body)
    tokens = list(map(unify_word, tokens))
    tokens = list(map(unify_word, tokens)) # some words fail filtering in the 1st time
    tokens = list(map(digit_filter, tokens)) 
    tokens = list(map(unify_word_meaning, tokens))
    tokens = [t for t in tokens if t not in stopWords and t != ""]
    return(tokens)


def value2int(y, clusters=2):
    label = np.copy(y)
    label[y < np.percentile(y, 100 / clusters)] = 0
    for i in range(1, clusters):
        label[y > np.percentile(y, 100 * i / clusters)] = i
    return label

def value2int_simple(y):
    label = np.copy(y)
    label[y < 0] = 0
    label[y >= 0] = 1
    return label


def model_eval(net, data_loader, if_print=1):
    net.eval()
    correct = 0
    total = 0
    for cnt, (images, labels) in enumerate(data_loader):
        images, labels = Variable(images), Variable(labels)
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        outputs = net.forward(images)
        prediction = outputs.data.max(1)[1]
        correct += prediction.eq(labels.data).sum().item()
    print('\nTest set: Accuracy: {:0.2f}%'.format(100.0 * correct / len(data_loader.dataset)))
