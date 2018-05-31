#!/usr/bin/env python3

import re
import time
import datetime
import numpy as np
from urllib.request import urlopen

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


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

wordnet = WordNetLemmatizer()
def unify_word(word):  # went -> go, apples -> apple, BIG -> big
    """unify verb tense and noun singular"""
    ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
    for wt in [ADJ, ADJ_SAT, ADV, NOUN, VERB]:
        try:
            word = wordnet.lemmatize(word, pos=wt)
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
