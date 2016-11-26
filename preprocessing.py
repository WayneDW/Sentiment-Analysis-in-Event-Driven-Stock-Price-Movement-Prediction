import os
import en
import nltk
import json
import numpy as np
import operator
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import reuters

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


def gen_financial_top_words(maxN=20000): # generate corpus based on Reuters news
    if not os.path.isfile('./input/topWords.json'):
        wordCnt = {}
        for field in reuters.fileids():
            for word in reuters.words(field):
                word = unify_word(word)
                if word in nltk.corpus.stopwords.words('english'):
                    continue
                wordCnt[word] = wordCnt.get(word, 0) + 1
        sorted_wordCnt = sorted(wordCnt.items(), key=operator.itemgetter(1), reverse=True)
        with open('./input/topWords.json', 'w') as fout: json.dump(sorted_wordCnt[:maxN], fout, indent=4)
    else: return

def unify_word(word):
    try: word = en.verb.present(word) # unify tense
    except: pass
    try: word = en.noun.singular(word) # unify noun
    except: pass
    return word.lower()


def build_FeatureMatrix(n_vocab=2000):
    if not os.path.isfile('./input/topWords.json'):
        gen_financial_top_words()
    with open('./input/topWords.json') as data_file:    
        topWords = json.load(data_file)

    with open('./input/stockPrices.json') as data_file:    
        priceDt = json.load(data_file)
    loc = './input/'
    input_files = [f for f in os.listdir(loc) if f.startswith('news_')]
    sentences = []
    word2idx = {'START': 0, 'END': 1}
    idx2word = ['START', 'END']
    current_idx = 2
    word_idx_count = {0: float('inf'), 1: float('inf')}

    labels = []
    for file in input_files:
        for line in open(loc + file):
            line = line.strip().split(',')
            if len(line) != 5: continue
            ticker, name, day, headline, body = line
            if ticker not in priceDt: print "??"; continue
            if day not in priceDt[ticker]: continue

            tokens = nltk.word_tokenize(headline) + nltk.word_tokenize(body)
            tokens = [unify_word(t) for t in tokens]
            for t in tokens:
                if t in nltk.corpus.stopwords.words('english') or t not in topWords:
                    tokens.remove(t)
            for t in tokens:
                if t not in word2idx:
                    word2idx[t] = current_idx
                    idx2word.append(t)
                    current_idx += 1
                idx = word2idx[t]
                word_idx_count[idx] = word_idx_count.get(idx, 0) + 1
            sentence_by_idx = [word2idx[t] for t in tokens]
            sentences.append(sentence_by_idx)
            labels.append(round(priceDt[ticker][day], 6))


    # restrict vocabulary size
    sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
    word2idx_small = {}
    new_idx = 0
    idx_new_idx_map = {}
    for idx, count in sorted_word_idx_count[:n_vocab]:
        word = idx2word[idx]
        #print word, count
        word2idx_small[word] = new_idx
        idx_new_idx_map[idx] = new_idx
        new_idx += 1
    # let 'unknown' be the last token
    word2idx_small['UNKNOWN'] = new_idx 
    unknown = new_idx

    # map full dict to truncated dict
    sentences_small = []
    new_label = []
    for sentence, label in zip(sentences, labels):
        if len(sentence) > 1:
            new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]
            sentences_small.append(new_sentence)
            new_label.append(label)

    max_words = 40
    truncated_small = np.matrix(sequence.pad_sequences(sentences_small, maxlen=max_words)).astype('int')
    print truncated_small
    new_label = np.matrix(new_label)
    new_label[new_label>0] = 1
    new_label[new_label<=0] = 0
    featureMatrix = np.concatenate((truncated_small, new_label.T), axis=1)
    print featureMatrix
    np.savetxt('./input/featureMatrix.csv', featureMatrix, delimiter=',', fmt="%d")
    # return truncated_small, new_label





if __name__ == '__main__':
    build_FeatureMatrix()



'''
Build word2idx vector based on dataset from Reuters

'''