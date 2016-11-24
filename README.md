# Sentiment Analysis in Stock Movement Prediction
Use NLP technique to predict stock price movement based on news from bloomberg and reuters



How to install scrapy
```
https://doc.scrapy.org/en/latest/intro/install.html
http://askubuntu.com/questions/675876/cant-install-some-packages-depends-x-but-x-is-to-be-installed
```

## 1. Data Preprocessing

Download the ticker list from NASDAQ

```python
./getList.py 20  # get the top N% marketcap companies, save to ./input/tickerList.csv
```

Use BeautifulSoup to crawl news headlines from Bloomberg and Reuters

```python
./crawler_bloomberg.py  # save to ./input/news_bloomberg.csv
./crawler_reuters.py  # save to ./input/news_reuters.csv
```

Use Yahoo Finance API to crawl historical stock prices

```python
./crawler_stockPrices.py # save to 
```

Correlate the stock movement with the associated news

## 2. Extract relations

Use Open IE to extract relations

## 3. Word Embeddings

## 4. Training and Model Selection

## 5. Prediction and analysis
