# Sentiment Analysis in Stock Movement Prediction
Use NLP technique to predict stock price movement based on news from bloomberg and reuters


1. data
2. use NLTK to remove stopwords, unify tense, puntuation
3. get score basd on tf-idf
4. dimension reduction
5. correlate the stock price and generate feature matrix
6. apply a neural network model to train
7. analyze


## Crawl data

1. Use [RCV](http://scikit-learn.org/stable/datasets/rcv1.html) in sklearn to get all the news contents to create our corpus. Remember to delete stop words, punctuation, etc. see detail in [tfidf-tsne](https://github.com/lazyprogrammer/machine_learning_examples/blob/master/nlp_class2/tfidf_tsne.py), on how to construct basic word2idx matrix, see [util.get_wikipedia_data](https://github.com/lazyprogrammer/machine_learning_examples/blob/master/rnn_class/util.py)

Download the ticker list from [NASDAQ](http://www.nasdaq.com/screening/companies-by-industry.aspx)

```python
./getList.py 20  # get the top N% marketcap companies
```

Use BeautifulSoup to crawl news headlines from [Bloomberg](http://www.bloomberg.com/search?query=goog&sort=time:desc) and [Reuters](http://www.reuters.com/finance/stocks/overview?symbol=FB.O)

```python
./crawler_bloomberg.py 
./crawler_reuters.py 
```

Use Yahoo Finance [API](https://pypi.python.org/pypi/yahoo-finance/1.1.4) to crawl historical stock prices

```python
./crawler_stockPrices.py
```

Correlate the stock movement with the associated news

## Data Preprocessing

lower case, remove punctuation, get rid of stop words using [NLTK](http://www.nltk.org/), unify tense using [en](https://www.nodebox.net/code/index.php/Linguistics#verb_conjugation)

## Extract relations

Use Open IE to extract relations

## Word Embeddings

## Training and Model Selection

## Prediction and analysis


## Issues
1. remove_punctuation() handles middle name (e.g., P.F -> pf)
