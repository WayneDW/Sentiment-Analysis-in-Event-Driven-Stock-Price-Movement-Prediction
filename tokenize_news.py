#!/usr/bin/env python3
import os
import operator
import argparse

import json
import numpy as np
import nltk

import util


""" Use pretrained word vector to generate our target features
Required input data:
./input/stopWords
./input/stockReturns.json
./input/news/*/*

Output file name: 
input/featureMatrix_train
input/featureMatrix_test 
input/word2idx"""

# credit: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/nlp_class2

def tokenize(news_file, price_file, stopWords_file, output, output_wd2idx, sen_len, term_type, n_vocab, mtype):
    # load price data
    with open(price_file) as file:
        print("Loading price info ... " + mtype)
        priceDt = json.load(file)[term_type]

    testDates = util.dateGenerator(90) # the most recent days are used for testing
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
    os.system('cat ./input/news/*/* > ./input/news_reuters.csv')
    with open(news_file) as f:
        for num, line in enumerate(f):
            line = line.strip().split(',')
            if len(line) not in [6, 7]:
                continue
            if len(line) == 6:
                ticker, name, day, headline, body, newsType = line
            else:
                ticker, name, day, headline, body, newsType, suggestion = line
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
            content = headline + ' ' + body
            content = content.replace("-", " ") 
            tokens = util.tokenize_news(content, stopWords)

            for t in tokens:
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
    total_num, cdf = 0.0, 0.0

    for idx, count in sorted_word_idx_count[:n_vocab]:
        if count == "inf" or count == float('inf'):
            continue
        total_num += count
    for idx, count in sorted_word_idx_count[:n_vocab]:
        word = idx2word[idx]
        if count == "inf" or count == float('inf'):
            continue
        cdf += (count * 1.0 / (total_num * 1.0))
        print(word, count, str(cdf)[:5])
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
            if len(new_sentence) > sen_len:
                new_sentence = new_sentence[:sen_len]
            else:
                new_sentence = new_sentence + [1] * (sen_len - len(new_sentence))
            new_sentence.append(labels[num])
            features.append(new_sentence)

    features = np.matrix(features)
    print(features.shape)


    with open(output_wd2idx, 'w') as fp:
        json.dump(word2idx_small, fp)

    with open(output + mtype, 'a+') as file:
        np.savetxt(file, features, fmt="%s")

def main():
    news_file = "./input/news_reuters.csv"
    stopWords_file = "./input/stopWords"
    price_file = "./input/stockReturns.json"

    
    output = './input/featureMatrix_'
    output_wd2idx = "./input/word2idx"

    parser = argparse.ArgumentParser(description='Tokenize Reuters news')
    parser.add_argument('-vocabs', type=int, default=6000, help='total number of vocabularies [default: 1000]')
    parser.add_argument('-words', type=int, default=40, help='max number of words in a sentence [default: 20]')
    parser.add_argument('-term', type=str, default='short', help='return type [short mid long] [default: short]')
    args = parser.parse_args()

    tokenize(news_file, price_file, stopWords_file, output, output_wd2idx, args.words, args.term, args.vocabs, 'train')
    tokenize(news_file, price_file, stopWords_file, output, output_wd2idx, args.words, args.term, args.vocabs, 'test')


if __name__ == "__main__":
    main()
