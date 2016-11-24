# Sentiment Analysis in Stock Movement Prediction
Use NLP technique to predict stock price movement based on news from bloomberg and reuters





## Crawl data

Download the ticker list from [NASDAQ](http://www.nasdaq.com/screening/companies-by-industry.aspx)

```python
./getList.py 20  # get the top N% marketcap companies
```

Use BeautifulSoup to crawl news headlines from [Bloomberg](http://www.bloomberg.com/search?query=goog&sort=time:desc) and Reuters

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
