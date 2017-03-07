#!/usr/bin/python
import json
import os
import en
import nltk
import numpy as np
import util

# Use pretrained word vector to generate our target features
# required input data:
# ./input/glove.xB.xd.txt 
# ./input/stopWords
# ./input/stockReturns.json
# ./input/news_reuters.csv
# ./input/featureMatrix

# output file name: 
# input/featureMatrix_train
# input/featureMatrix_test


def wordVec(glove_file):
    wordDict = {}
    with open(glove_file) as f:
        print("Loading word vector ...")
        for line in f:
            line = line.strip().split(' ')
            key, values = line[0], map(float, line[1:])
            wordDict[key] = values
    return wordDict, len(values) # return word vector and word vector dimension


def gen_FeatureMatrix(news_file, price_file, stopWords_file, output, wordDict, dim_wordVec, sentense_len, term_type, mtype):
    with open(price_file) as file:
        print("Loading price info ...")
        priceDt = json.load(file)[term_type]
    cnt = 0
    testDates = util.dateGenerator(300)
    os.system('rm ' + output + mtype)

    stopWords = set()
    with open(stopWords_file) as file:
        for word in file:
            stopWords.add(word.strip())

    with open(news_file) as f:
        for line in f:
            line = line.strip().split(',')
            if len(line) != 6: continue
            '''
            newsType: [topStory, normal]
            '''
            ticker, name, day, headline, body, newsType = line
            if newsType != 'topStory': continue # skip normal news
            if ticker not in priceDt: continue # skip if no corresponding company found
            if day not in priceDt[ticker]: continue # skip if no corresponding date found
            cnt += 1
            # if cnt > 20: continue
            if cnt % 1000 == 0: print("%sing samples %d" % (mtype, cnt))
            if mtype == "test" and day not in testDates: continue
            if mtype == "train" and day in testDates: continue
            # 2.1 tokenize sentense, check if the word belongs to the top words, unify the format of words
            #headline = headline.encode('utf-8')
            #body = body.encode('utf-8')

            tokens = nltk.word_tokenize(headline) # + nltk.word_tokenize(body)
            tokens = map(util.unify_word, tokens)

            # build feature and label
            feature = np.zeros([0, dim_wordVec])
            featureNone = True
            for t in tokens:
                # if t in stopWords: continue
                if t not in wordDict: continue
                featureNone = False
                feature = np.vstack((feature, np.matrix(wordDict[t])))
            if featureNone: continue # feature is empty, continue

            feature = util.padding(feature, sentense_len)
            label = round(priceDt[ticker][day], 6)

            with open(output + mtype, 'a+') as file:
                np.savetxt(file, np.hstack((feature, np.matrix(label))), fmt='%.5f')

def main():
    glove_file = "./input/glove.6B.100d.txt"
    news_file = "./input/news_reuters.csv"
    stopWords_file = "./input/stopWords"
    price_file = "./input/stockReturns.json"
    output = './input/featureMatrix_'
    sentense_len = 20
    term_type = 'short'
    wordDict, dim_wordVec = wordVec(glove_file)
    gen_FeatureMatrix(news_file, price_file, stopWords_file, output, wordDict, dim_wordVec, sentense_len, term_type, 'train')
    gen_FeatureMatrix(news_file, price_file, stopWords_file, output, wordDict, dim_wordVec, sentense_len, term_type, 'test')


if __name__ == "__main__":
    main()
