#!/usr/bin/env python3
import os
import operator

import json
import numpy as np
import nltk
from nltk.corpus import reuters

import util


""" Use pretrained word vector to generate our target features
Required input data:
./input/stopWords
./input/stockReturns.json
./input/news_reuters.csv

Output file name: 
input/featureMatrix_train
input/featureMatrix_test """

# credit: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/nlp_class2

def tokenize(news_file, price_file, stopWords_file, output, sentense_len, term_type, n_vocab, mtype):
    # load price data
    with open(price_file) as file:
        print("Loading price info ...")
        priceDt = json.load(file)[term_type]

    testDates = util.dateGenerator(1) # the most recent days are used for testing
    os.system('rm ' + output + mtype)

    # load stop words
    stopWords = set()
    with open(stopWords_file) as file:
        for word in file:
            stopWords.add(word.strip())

    # build feature matrix
    word2idx = {'START': 0, 'END': 1}
    idx2word = ['START', 'END']
    current_idx = 2
    word_idx_count = {0: float('inf'), 1: float('inf')}
    sentences, labels = [], []
    with open(news_file) as f:
        for num, line in enumerate(f):
            line = line.strip().split(',')
            if len(line) != 6:
                continue
            ticker, name, day, headline, body, newsType = line
            
            if newsType != 'topStory': # newsType: [topStory, normal]
                continue # skip normal news
            
            if ticker not in priceDt: 
                continue # skip if no corresponding company found
            if day not in priceDt[ticker]: 
                continue # skip if no corresponding date found

            if num % 10000 == 0: 
                print("%sing samples %d" % (mtype, num))
            if mtype == "test" and day not in testDates: 
                continue
            if mtype == "train" and day in testDates: 
                continue

            tokens = nltk.word_tokenize(headline) + nltk.word_tokenize(body)
            tokens = list(map(util.unify_word, tokens))
            tokens = list(map(util.unify_word, tokens))

            for t in tokens:
                if t in stopWords:
                    continue
                if t not in word2idx:
                    word2idx[t] = current_idx
                    idx2word.append(t)
                    current_idx += 1
                idx = word2idx[t]
                word_idx_count[idx] = word_idx_count.get(idx, 0) + 1
            sentence_by_idx = [word2idx[t] for t in tokens if t not in stopWords]
            sentences.append(sentence_by_idx)
            labels.append(round(priceDt[ticker][day], 6))

    # restrict vocab size
    sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
    word2idx_small = {}
    new_idx = 0
    idx_new_idx_map = {}
    for idx, count in sorted_word_idx_count[:n_vocab]:
        word = idx2word[idx]
        print(word, count)
        word2idx_small[word] = new_idx
        idx_new_idx_map[idx] = new_idx
        new_idx += 1
    # let 'unknown' be the last token
    word2idx_small['UNKNOWN'] = new_idx 
    unknown = new_idx

    # map old idx to new idx
    features = [] # shorter sentence idx
    for num, sentence in enumerate(sentences):
        if len(sentence) > 1:
            new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]
            # padding
            if len(new_sentence) > sentense_len:
                new_sentence = new_sentence[:sentense_len]
            else:
                new_sentence = new_sentence + [1] * (sentense_len - len(new_sentence))
            new_sentence.append(labels[num])
            features.append(new_sentence)

    features = np.matrix(features)
    print(features.shape)

    with open(output + mtype, 'a+') as file:
        np.savetxt(file, features, fmt="%s")

def main():
    news_file = "./input/news_reuters.csv"
    stopWords_file = "./input/stopWords"
    price_file = "./input/stockReturns.json"
    output = './input/featureMatrix_'

    n_vocab = 10000
    sentense_len = 30
    # you can choose short mid long
    term_type = 'short'
    tokenize(news_file, price_file, stopWords_file, output, sentense_len, term_type, n_vocab, 'train')
    tokenize(news_file, price_file, stopWords_file, output, sentense_len, term_type, n_vocab, 'test')


if __name__ == "__main__":
    main()
