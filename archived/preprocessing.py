#!/usr/bin/python
import os
import en
import nltk
import json
import numpy as np
import operator
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import reuters
from keras.preprocessing import sequence



def gen_financial_top_words(maxN=40000): # generate corpus based on Reuters news
    if not os.path.isfile('./input/topWords.json'):
        wordCnt = {}
        for field in reuters.fileids():
            for word in reuters.words(field):
                word = unify_word(word)
                if word in nltk.corpus.stopwords.words('english'):
                    continue
                wordCnt[word] = wordCnt.get(word, 0) + 1

        sorted_wordCnt = sorted(wordCnt.items(), key=operator.itemgetter(1), reverse=True)
        wordCnt = {} # reset wordCnt
        for i in sorted_wordCnt[:maxN]: wordCnt[i[0]] = i[1] # convert list to dict
        with open('./input/topWords.json', 'w') as fout: json.dump(wordCnt, fout, indent=4)
    else: return

def unify_word(word):
    try: word = en.verb.present(word) # unify tense
    except: pass
    try: word = en.noun.singular(word) # unify noun
    except: pass
    return word.lower()

def build_FeatureMatrix(max_words=20, n_vocab=2000):
    if not os.path.isfile('./input/topWords.json'):
        gen_financial_top_words()
    with open('./input/topWords.json') as data_file:    
        topWords = json.load(data_file)

    with open('./input/stockPrices.json') as data_file:    
        priceDt = json.load(data_file)
    loc = './input/'
    input_files = [f for f in os.listdir(loc) if f.startswith('news_reuters.csv')]
    sentences = []
    word2idx = {'START': 0, 'END': 1}
    idx2word = ['START', 'END']
    current_idx = 2
    word_idx_cnt = {0: float('inf'), 1: float('inf')}
    labels = []
    dp = {} # only consider one news for a company everyday
    cnt = 0
    for file in input_files:
        for line in open(loc + file):
            line = line.strip().split(',')
            if len(line) != 5: continue
            ticker, name, day, headline, body = line
            
            if ticker not in priceDt: continue
            if day not in priceDt[ticker]: continue
            # avoid repeating news
            if ticker not in dp: dp[ticker] = set()
            if day in dp[ticker]: continue
            dp[ticker].add(day)
            print(cnt, ticker); cnt += 1
            tokens = nltk.word_tokenize(headline) + nltk.word_tokenize(body)
            tokens = [t for t in tokens if t in topWords]
            for t in tokens:
                if t not in word2idx:
                    word2idx[t] = current_idx
                    idx2word.append(t)
                    current_idx += 1
                idx = word2idx[t]
                word_idx_cnt[idx] = word_idx_cnt.get(idx, 0) + 1
            sentence_by_idx = [word2idx[t] for t in tokens]
            sentences.append(sentence_by_idx)
            labels.append(round(priceDt[ticker][day], 6))


    # restrict vocabulary size
    sorted_word_idx_cnt = sorted(word_idx_cnt.items(), key=operator.itemgetter(1), reverse=True)
    word2idx_small = {}
    new_idx = 0
    idx_new_map = {}
    for idx, count in sorted_word_idx_cnt[:n_vocab]:
        word = idx2word[idx]
        word2idx_small[word] = new_idx
        idx_new_map[idx] = new_idx
        new_idx += 1
    # let 'unknown' be the last token
    word2idx_small['UNKNOWN'] = new_idx 
    unknown = new_idx

    # map full dict to truncated dict
    sentences_small = []
    new_label = []
    for sentence, label in zip(sentences, labels):
        if len(sentence) > 1:
            new_sentence = [idx_new_map[idx] if idx in idx_new_map else unknown for idx in sentence]
            sentences_small.append(new_sentence)
            new_label.append(label)

    
    truncated_small = np.matrix(sequence.pad_sequences(sentences_small, maxlen=max_words))
    truncated_small = truncated_small.astype('int').astype('str')
    new_label = np.matrix(new_label).astype('str')

    featureMatrix = np.concatenate((truncated_small, new_label.T), axis=1)
    np.savetxt('./input/featureMatrix.csv', featureMatrix, fmt="%s")



if __name__ == '__main__':
    build_FeatureMatrix(max_words=20)

