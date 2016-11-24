# Sentiment Analysis in Stock Movement Prediction
Use NLP technique to predict stock price movement based on news from bloomberg and reuters





## 1. Data Preprocessing

Download the ticker list from NASDAQ

```python
./getList.py 20  # get the top N% marketcap companies
```

Use BeautifulSoup to crawl news headlines from Bloomberg and Reuters

```python
./crawler_bloomberg.py 
./crawler_reuters.py 
```

Use Yahoo Finance API to crawl historical stock prices

```python
./crawler_stockPrices.py
```

Correlate the stock movement with the associated news

## 2. Extract relations

Use Open IE to extract relations

## 3. Word Embeddings

## 4. Training and Model Selection

## 5. Prediction and analysis
