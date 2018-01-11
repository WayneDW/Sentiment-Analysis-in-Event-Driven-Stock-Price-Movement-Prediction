# Adopted from https://github.com/lazyprogrammer/machine_learning_examples/blob/master/nlp_class2/tfidf_tsne.py
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from datetime import datetime


# import os
# import sys
# sys.path.append(os.path.abspath('..'))
from utils import get_wikipedia_data, find_analogies, get_news_data_with_price
# from util import find_analogies
from sklearn.feature_extraction.text import TfidfTransformer

def tsne_on_wikipedia():
    sentences, word2idx = get_wikipedia_data('file', 5000, by_paragraph=True)
    with open('w2v_word2idx.json', 'w') as f:
        json.dump(word2idx, f)

    # build term document matrix
    V = len(word2idx)
    N = len(sentences)
    print V, N

    # create raw counts first
    A = np.zeros((V, N))
    j = 0
    for sentence in sentences:
        for i in sentence:
            A[i,j] += 1
        j += 1
    print 'finished getting raw counts'

    transformer = TfidfTransformer()
    A = transformer.fit_transform(A)
    A = A.toarray()

    idx2word = {v:k for k, v in word2idx.iteritems()}

    # plot the data in 2-D
    tsne = TSNE()
    Z = tsne.fit_transform(A)
    print 'Z.shape:', Z.shape
    plt.scatter(Z[:,0], Z[:,1])
    for i in xrange(V):
        try:
            plt.annotate(s=idx2word[i].encode('utf8'), xy=(Z[i,0], Z[i,1]))
        except:
            print 'bad string:', idx2word[i]
    plt.show()

    We = Z
    # find_analogies('king', 'man', 'woman', We, word2idx)
    find_analogies('france', 'paris', 'london', We, word2idx)
    find_analogies('france', 'paris', 'rome', We, word2idx)
    find_analogies('paris', 'france', 'italy', We, word2idx)    

def tsne_on_news():
    get_news_data_with_price()


if __name__ == '__main__':
    tsne_on_news()