# Sentiment Analysis in Stock Movement Prediction
Use NLP technique to predict stock price movement based on news from Reuters

1. Data Collection

  1.1 get the whole ticker list
  
  1.2 crawl news from Reuters using BeautifulSoup
  
  1.3 crawl prices using Yahoo Finance API
  
2. Applied GloVe to train a dense word vector from Reuters corpus in NLTK
3. Feature Engineering

  3.1 Extract feature using feature hashing
  
  3.2 Remove punctuations, unify tense, singular & plural
  
  3.3 Pad word senquence (essentially a matrix)
  
4. Train a ConvoNet to predict the stock price movement.
5. The result shows a 15% percent improve on the validation set, and 1-2% percent improve on the test set


### 1. Data Collection


#### 1.1 Download the ticker list from [NASDAQ](http://www.nasdaq.com/screening/companies-by-industry.aspx)

```python
./crawler_allTickers.py 20  # keep the top e.g. 20% marketcap companies
```

#### 1.2 Use BeautifulSoup to crawl news headlines from [Bloomberg](http://www.bloomberg.com/search?query=goog&sort=time:desc) and [Reuters](http://www.reuters.com/finance/stocks/overview?symbol=FB.O)

```python
./crawler_bloomberg.py # Bloomberg news is not that correlated, ignore this data at this moment
./crawler_reuters.py # much more valueable, despite with a much smaller size
```

#### 1.3 Use [Yahoo Finance API](https://pypi.python.org/pypi/yahoo-finance/1.1.4) to crawl historical stock prices

```python
./crawler_stockPrices.py
```

### 2. Word Embedding

Applied [GloVe](https://github.com/lazyprogrammer/machine_learning_examples/blob/master/nlp_class2/glove.py) to train a dense word vector from Reuters corpus in NLTK

```python
./word_embedding.py
```

### 3. Feature Engineering

Unify the word format, project word in a sentence to the word vector, so every sentence results in a matrix.
Lower case, remove punctuation, get rid of stop words using [NLTK](http://www.nltk.org/) (remark here, I didn't use it in the latest version), unify tense and singular & plural using [en](https://www.nodebox.net/code/index.php/Linguistics#verb_conjugation)

Most importantly, we should seperate test set away from training+validation test, otherwise we would get a too optimistic result.

```python
./genFeatureMatrix.py
```

### 4. Train a ConvoNet to predict the stock price movement. 

For the sake of simplicity, I just applied a ConvoNet in [Keras](http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/), the detail operations in text data is slighly differnt from the image, we can use the structure from [FIgure 1 in Yoon Kim's paper](http://www.aclweb.org/anthology/D14-1181)

```python
./model_cnn.py
```

### 5. Prediction and analysis

As shown in the result, the performance has some extent improvement. The result from validation set is way higher than the test result, which may result in a not sufficient sample number.

One remark here is that the dropout ratio set as 40% or 50% can help improve the testing result a little bit.

./output/result_glove_cnn_128filters_50dropout_1hiddenLayer64nodes_binaryClassification

### 6. Future work

From the [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1331573) by Tim Loughran and Bill McDonald, some words have strong indication of positive and negative effects, we may need to dig into these words to find more information. A very simple but interest example can be found in [Financial Sentiment Analysis part1](http://francescopochetti.com/scrapying-around-web/), [part2](http://francescopochetti.com/financial-blogs-sentiment-analysis-part-crawling-web/)

Another idea is to reconstruct the negative words, like 'not good' -> 'notgood'


## Issues
1. remove_punctuation() handles middle name (e.g., P.F -> pf)

## References:
1. [Keras predict sentiment-movie-reviews using deep learning](http://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/)
2. [Keras sequence-classification-lstm-recurrent-neural-networks](http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/)
3. [tf-idf + t-sne](https://github.com/lazyprogrammer/machine_learning_examples/blob/master/nlp_class2/tfidf_tsne.py)
4. [IMPLEMENTING A CNN FOR TEXT CLASSIFICATION IN TENSORFLOW](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
5. [Implementation of CNN in sequence classification](https://github.com/dennybritz/cnn-text-classification-tf)
6. [Convolutional Neural Networks for Sentence Classification](http://www.aclweb.org/anthology/D14-1181)
7. [GloVe: Global Vectors for Word Representation](http://www-nlp.stanford.edu/pubs/glove.pdf)
8. Tim Loughran and Bill McDonald, 2011, “When is a Liability not a Liability?  Textual Analysis, Dictionaries, and 10-Ks,” Journal of Finance, 66:1, 35-65.
