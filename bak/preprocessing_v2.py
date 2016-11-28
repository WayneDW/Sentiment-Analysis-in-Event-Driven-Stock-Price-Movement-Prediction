#!/usr/bin/python
import os
import en
import nltk
import json
import numpy as np
import operator
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import reuters
from keras.preprocessing import sequence
from crawler_reuters import news_Reuters



# def gen_financial_top_words(maxN=40000): # generate corpus based on Reuters news
#     if not os.path.isfile('./input/topWords.json'):
#         wordCnt = {}
#         for field in reuters.fileids():
#             for word in reuters.words(field):
#                 word = unify_word(word)
#                 if word in nltk.corpus.stopwords.words('english'):
#                     continue
#                 wordCnt[word] = wordCnt.get(word, 0) + 1

#         sorted_wordCnt = sorted(wordCnt.items(), key=operator.itemgetter(1), reverse=True)
#         wordCnt = {} # reset wordCnt
#         for i in sorted_wordCnt[:maxN]: wordCnt[i[0]] = i[1] # convert list to dict
#         with open('./input/topWords.json', 'w') as fout: json.dump(wordCnt, fout, indent=4)
#     else: return

def unify_word(word): # went -> go, apples -> apple, BIG -> big
    try: word = en.verb.present(word) # unify tense
    except: pass
    try: word = en.noun.singular(word) # unify noun
    except: pass
    return word.lower()

def dateGenerator(numdays): # generate N days until now, eg [20151231, 20151230]
    base = datetime.datetime.today()
    date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
    for i in range(len(date_list)): date_list[i] = date_list[i].strftime("%Y%m%d")
    return set(date_list)


'''
The following function is a little complicated.
It consists of the following steps
1, load top words dictionary, load prices data to make correlation
2, build feature matrix for training data
    2.1 tokenize sentense, check if the word belongs to the top words, unify the format of words
    2.2 create word2idx/idx2word list, and a list to count the occurence of words
    2.3 concatenate multi-news into a single one if they happened at the same day
    2.4 limit the vocabulary size to e.g. 2000, and let the unkown words as the last one
    2.5 map full dict to truncated dict, pad the sequence to the same length, done
3, project the test feature in the word2idx for the traning data
'''
def build_FeatureMatrix(max_words=60, n_vocab=2000):
    # step 1, load top words dictionary, load prices data to make correlation
    if not os.path.isfile('./input/topWords.json'):
        gen_financial_top_words()
    # with open('./input/topWords.json') as data_file:    
    #     topWords = json.load(data_file)

    with open('./input/stockPrices.json') as data_file:    
        priceDt = json.load(data_file)
    # step 2, build feature matrix for training data
    loc = './input/'
    input_files = [f for f in os.listdir(loc) if f.startswith('news_reuters.csv')]
    word2idx = {'START': 0, 'END': 1}
    idx2word = ['START', 'END']
    current_idx = 2
    word_idx_cnt = {0: float('inf'), 1: float('inf')}
    dp = {} # only consider one news for a company everyday
    cnt = 0
    testDates = dateGenerator(100)
    stopWords = set(nltk.corpus.stopwords.words('english'))
    for file in input_files:
        for line in open(loc + file):
            line = line.strip().split(',')
            if len(line) != 5: continue
            ticker, name, day, headline, body = line
            if ticker not in priceDt: continue # skip if no corresponding company found
            if day not in priceDt[ticker]: continue # skip if no corresponding date found
            # # avoid repeating news
            if ticker not in dp: dp[ticker] = {}
            if day not in dp[ticker]: dp[ticker][day] = {'feature':[], 'label':[]}
            # if ticker not in dp: dp[ticker] = set()
            # if day in dp[ticker]: continue
            # dp[ticker].add(day)
            # 2.1 tokenize sentense, check if the word belongs to the top words, unify the format of words
            tokens = nltk.word_tokenize(headline) + nltk.word_tokenize(body)
            tokens = [unify_word(t) for t in tokens]
            tokens = [t for t in tokens if t in stopWords]
            #tokens = [t for t in tokens if t in topWords]
            # 2.2 create word2idx/idx2word list, and a list to count the occurence of words
            for t in tokens:
                if t not in word2idx:
                    word2idx[t] = current_idx
                    idx2word.append(t)
                    current_idx += 1
                idx = word2idx[t]
                word_idx_cnt[idx] = word_idx_cnt.get(idx, 0) + 1
            if day in testDates: continue # this step only considers training set
            sentence_by_idx = [word2idx[t] for t in tokens]
            print("training", cnt, ticker); cnt += 1
            #sentences.append(sentence_by_idx)
            dp[ticker][day]['feature'].append(sentence_by_idx)
            dp[ticker][day]['label'] = round(priceDt[ticker][day], 6)

    # 2.3 concatenate multi-news into a single one if they happened at the same day
    sentences, labels, sentenceLen = [], [], []
    for ticker in dp:
        for day in dp[ticker]:
            res = []
            for i in dp[ticker][day]['feature']: res += i
            sentenceLen.append(len(res))
            sentences.append(res)
            labels.append(dp[ticker][day]['label'])

    sentenceLen = np.array(sentenceLen)

    for percent in [50, 70, 80, 90, 95, 99]:
        print("Sentence length %d%% percentile: %d" % (percent, np.percentile(sentenceLen, percent)))

    # 2.4 limit the vocabulary size to e.g. 2000, and let the unkown words as the last one
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

    # 2.5 map full dict to truncated dict, pad the sequence to the same length, done
    sentences_small = []
    new_label = []
    for sentence, label in zip(sentences, labels):
        if len(sentence) > 1:
            new_sentence = [idx_new_map[idx] if idx in idx_new_map else unknown for idx in sentence]
            sentences_small.append(new_sentence)
            new_label.append(label)

    pad_small = np.matrix(sequence.pad_sequences(sentences_small, maxlen=max_words))
    pad_small = pad_small.astype('int').astype('str')
    new_label = np.matrix(new_label).astype('str')

    featureMatrix = np.concatenate((pad_small, new_label.T), axis=1)
    np.savetxt('./input/featureMatrix.csv', featureMatrix, fmt="%s")

    # step 3, project the test feature in the word2idx for the traning data
    dp = {}; cnt = 0
    for file in input_files:
        for line in open(loc + file):
            line = line.strip().split(',')
            if len(line) != 5: continue
            ticker, name, day, headline, body = line
            if day not in testDates: continue # this step only considers test set
            if ticker not in priceDt: continue # continue if no corresponding prices information found
            if day not in priceDt[ticker]: continue
            # modify repeating news
            if ticker not in dp: dp[ticker] = {}
            if day not in dp[ticker]: dp[ticker][day] = {'feature':[], 'label':[]}
            cnt += 1
            tokens = nltk.word_tokenize(headline) + nltk.word_tokenize(body)
            tokens = [unify_word(t) for t in tokens]
            tokens = [t for t in tokens if t in stopWords]
            #tokens = [t for t in tokens if t in topWords]
            sentence_by_idx = [word2idx_small[t] for t in tokens if t in word2idx_small]
            dp[ticker][day]['feature'].append(sentence_by_idx)
            dp[ticker][day]['label'] = round(priceDt[ticker][day], 6)
    print("test", cnt)
    sentences_test, labels_test = [], []
    for ticker in dp:
        for day in dp[ticker]:
            res = []
            for i in dp[ticker][day]['feature']: res += i
            sentences_test.append(res)
            labels_test.append(dp[ticker][day]['label'])

    pad_test = np.matrix(sequence.pad_sequences(sentences_test, maxlen=max_words))
    pad_test = pad_test.astype('int').astype('str')
    labels_test = np.matrix(labels_test).astype('str')
    featureMatrix = np.concatenate((pad_test, labels_test.T), axis=1)
    np.savetxt('./input/featureMatrixTest.csv', featureMatrix, fmt="%s")



if __name__ == '__main__':
    build_FeatureMatrix(max_words=80)

