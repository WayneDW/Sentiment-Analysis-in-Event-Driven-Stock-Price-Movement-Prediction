# Sentiment Analysis for Event-Driven Stock Prediction
Use NLP method to predict stock price movement based on news from Reuters

1. Data Collection

  1.1 get the whole ticker list
  
  1.2 crawl news from Reuters using BeautifulSoup
  
  1.3 crawl prices using Yahoo Finance API
  
2. Applied GloVe to train a dense word vector from Reuters corpus in NLTK

  2.1 build the word-word co-occurrence matrix
  
  2.2 factorizing the weighted log of the co-occurrence matrix
  
3. Feature Engineering
  
  3.2 Unify word format: remove punctuations, unify tense, singular & plural
  
  3.2 Extract feature using feature hashing based on the trained word vector (step 2)
  
  3.3 Pad word senquence (essentially a matrix) to keep the same dimension
  
4. Trained a ConvNet to predict the stock price movement based on a reasonable parameter selection
5. The result shows a 15% percent improvement on the validation set, and 1-2% percent improve on the test set


### 1. Data Collection


#### 1.1 Download the ticker list from [NASDAQ](http://www.nasdaq.com/screening/companies-by-industry.aspx)

```python
./crawler_allTickers.py 20  # keep the top e.g. 20% marketcap companies
```

#### 1.2 Use BeautifulSoup to crawl news headlines from [Bloomberg](http://www.bloomberg.com/search?query=goog&sort=time:desc) and [Reuters](http://www.reuters.com/finance/stocks/overview?symbol=FB.O)

Suppose we find a news about Facebook on Dec.13, 2016 at reuters.com

![](./imgs/tar1.PNG)

We can use the following script to crawl it and format it to our local file

```python
./crawler_reuters.py # more precise than Bloomberg News
```

![](./imgs/tar2.PNG)

#### 1.3 Use [Yahoo Finance API](https://pypi.python.org/pypi/yahoo-finance/1.1.4) to crawl historical stock prices

```python
./crawler_stockPrices.py
```

### 2. Word Embedding

Applied GloVe to train a dense word vector from Reuters corpus in NLTK

```python
./embeddingWord.py
```

About the detail of the method, [link](http://www-nlp.stanford.edu/pubs/glove.pdf)

About the implementation of this method, [link](https://github.com/lazyprogrammer/machine_learning_examples/blob/master/nlp_class2/glove.py)

### 3. Feature Engineering

Unify the word format, project word in a sentence to the word vector, so every sentence results in a matrix.

A little more detail about word format: lower case, remove punctuation, get rid of stop words using [NLTK](http://www.nltk.org/) (remark here, I didn't use it in the latest version), unify tense and singular & plural using [en](https://www.nodebox.net/code/index.php/Linguistics#verb_conjugation)

Most importantly, we should seperate test set away from training+validation test, otherwise we would get a too optimistic result.

```python
./genFeatureMatrix.py
```

### 4. Train a ConvoNet to predict the stock price movement. 

For the sake of simplicity, I just applied a ConvoNet in [Keras](http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/), the detail operations in text data is slighly differnt from the image, we can use the architecture from [FIgure 1 in Yoon Kim's paper](http://www.aclweb.org/anthology/D14-1181)

```python
./model_cnn.py
```

### 5. Prediction and analysis

As shown in the result, the performance has some extent improvement. The result from validation set is way higher than the test result, which may result in a not sufficient sample number.

./output/result_glove_cnn_128filters_50dropout_1hiddenLayer64nodes_binaryClassification

One remark here is that the dropout ratio set as 40% or 50% can help improve the testing result a little bit.

### 6. Future work

From the [work](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1331573) by Tim Loughran and Bill McDonald, some words have strong indication of positive and negative effects in finance, we may need to dig into these words to find more information. A very simple but interest example can be found in [Financial Sentiment Analysis part1](http://francescopochetti.com/scrapying-around-web/), [part2](http://francescopochetti.com/financial-blogs-sentiment-analysis-part-crawling-web/)

Another idea is to reconstruct the negative words, like 'not good' -> 'notgood'

We have lots of data from Bloomberg, however the keyword may not be the corresponding news for the specific company, one way to solve that is to filter a list with target company names. Like facebook keyword may result in a much more correlated news than financial.



## Issues
1. remove_punctuation() handles middle name (e.g., P.F -> pf)

## References:

1. Yoon Kim, [Convolutional Neural Networks for Sentence Classification](http://www.aclweb.org/anthology/D14-1181), EMNLP, 2014
2. J Pennington, R Socher, CD Manning, [GloVe: Global Vectors for Word Representation](http://www-nlp.stanford.edu/pubs/glove.pdf), EMNLP, 2014
3. Tim Loughran and Bill McDonald, 2011, “When is a Liability not a Liability?  Textual Analysis, Dictionaries, and 10-Ks,” Journal of Finance, 66:1, 35-65.
4. H Lee, etc, [On the Importance of Text Analysis for Stock Price Prediction](http://nlp.stanford.edu/pubs/lrec2014-stock.pdf), LREC, 2014
5. Xiao Ding, [Deep Learning for Event-Driven Stock Prediction](http://ijcai.org/Proceedings/15/Papers/329.pdf), IJCAI2015
6. [IMPLEMENTING A CNN FOR TEXT CLASSIFICATION IN TENSORFLOW](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
7. [Keras predict sentiment-movie-reviews using deep learning](http://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/)
8. [Keras sequence-classification-lstm-recurrent-neural-networks](http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/)
9. [tf-idf + t-sne](https://github.com/lazyprogrammer/machine_learning_examples/blob/master/nlp_class2/tfidf_tsne.py)
10. [Implementation of CNN in sequence classification](https://github.com/dennybritz/cnn-text-classification-tf)
11. [Getting Started with Word2Vec and GloVe in Python](http://textminingonline.com/getting-started-with-word2vec-and-glove-in-python)
