# Adopted from https://github.com/lazyprogrammer/machine_learning_examples/blob/master/rnn_class/util.py
# Adopted form https://github.com/lazyprogrammer/machine_learning_examples/blob/master/nlp_class2/util.py
import numpy as np
import pandas as pd
import string
import os
import operator
from nltk import pos_tag, word_tokenize
from datetime import datetime

import time  # for debug

from nltk.corpus import stopwords
eng_stop = set(stopwords.words('english'))

def remove_punctuation(s):
    return s.translate(None, string.punctuation)

def my_tokenizer(s):
    s = remove_punctuation(s)
    s = s.lower() # downcase
    # remove stopwords
    return [i for i in s.split() if i not in eng_stop]

def get_wikipedia_data(filename, n_vocab, by_paragraph=False):
    prefix = './input/'
    # return variables
    sentences = []
    word2idx = {'START': 0, 'END': 1}
    idx2word = ['START', 'END']
    current_idx = 2
    word_idx_count = {0: float('inf'), 1: float('inf')}

    print "reading:", filename
    for line in open(prefix + filename):
        line = line.strip()
        # don't count headers, structured data, lists, etc...
        if line and line[0] not in ('[', '*', '-', '|', '=', '{', '}'):
            if by_paragraph:
                sentence_lines = [line]
            else:
                sentence_lines = line.split('. ')
            for sentence in sentence_lines:
                tokens = my_tokenizer(sentence)
                for t in tokens:
                    if t not in word2idx:
                        word2idx[t] = current_idx
                        idx2word.append(t)
                        current_idx += 1
                    idx = word2idx[t]
                    word_idx_count[idx] = word_idx_count.get(idx, 0) + 1
                sentence_by_idx = [word2idx[t] for t in tokens]
                sentences.append(sentence_by_idx)

    print '# of unique words: ', len(word2idx)

    # restrict vocab size
    sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
    word2idx_small = {}
    new_idx = 0
    idx_new_idx_map = {}
    for idx, count in sorted_word_idx_count[:n_vocab]:
        word = idx2word[idx]
        print word, count
        word2idx_small[word] = new_idx
        idx_new_idx_map[idx] = new_idx
        new_idx += 1
    # let 'unknown' be the last token
    word2idx_small['UNKNOWN'] = new_idx 
    unknown = new_idx

    assert('START' in word2idx_small)
    assert('END' in word2idx_small)
    # assert('king' in word2idx_small)
    # assert('queen' in word2idx_small)
    # assert('man' in word2idx_small)
    # assert('woman' in word2idx_small)

    # map old idx to new idx
    sentences_small = []
    for sentence in sentences:
        if len(sentence) > 1:
            new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]
            sentences_small.append(new_sentence)

    return sentences_small, word2idx_small


def find_analogies(w1, w2, w3, We, word2idx):
    king = We[word2idx[w1]]
    man = We[word2idx[w2]]
    woman = We[word2idx[w3]]
    v0 = king - man + woman

    def dist1(a, b):
        return np.linalg.norm(a - b)
    def dist2(a, b):
        return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

    for dist, name in [(dist1, 'Euclidean'), (dist2, 'cosine')]:
        min_dist = float('inf')
        best_word = ''
        for word, idx in word2idx.iteritems():
            if word not in (w1, w2, w3):
                v1 = We[idx]
                d = dist(v0, v1)
                if d < min_dist:
                    min_dist = d
                    best_word = word
        print "closest match by", name, "distance:", best_word
        print w1, "-", w2, "=", best_word, "-", w3    


def get_news_data_with_price(filename, prefix='./input/'):
    df = pd.read_csv(prefix+filename, header=None)
    # use line numbers to check if data is filtered or not
    lineNo = df.shape[0]
    filtered_filename = './filtered/'+ str(lineNo) + '_' + filename
    print 'try to read', filtered_filename
    if os.path.isfile(filtered_filename):
        df = pd.read_csv(filtered_filename)
        data = df.as_matrix()
        X = data[:, :-1]
        Y = data[:, -1]
        print 'Done!'
        return X, Y
    
    # save if new
    print "filtered data doesn't exist, filter and save"
    df.columns = ['Ticker', 'Comp_name', 'Date', 'Title', 'Summary']
    print df.head()



def main():
    # get_wikipedia_data('file', 5000)
    get_news_data_with_price('news_bloomberg_part0.csv')


if __name__ == '__main__':
    main()
