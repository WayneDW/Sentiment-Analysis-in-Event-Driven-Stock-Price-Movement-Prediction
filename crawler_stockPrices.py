#!/usr/bin/python
import sys
import os
import json
import time
import numpy as np
import random
from math import log
from yahoo_finance import Share
from pprint import pprint

def stock_Prices():
    # exit if the output already existed
    if os.path.isfile('./input/stockPrices.json'):
        sys.exit("Prices data already existed!")

    priceSet = []
    fin = open('./input/tickerList.csv')
    for num, line in enumerate(fin):
        line = line.strip().split(',')
        ticker, name, exchange, MarketCap = line
        try:
            yahoo = Share(ticker)
            time.sleep(np.random.poisson(3))
            prices = yahoo.get_historical('2009-01-01', '2020-01-01')
            for i in range(len(prices)):
                    prices[i]['Return'] = round(log(float(prices[i]['Close']) / float(prices[i]['Open'])), 6)
                    del prices[i]['High'], prices[i]['Low'], prices[i]['Open'], prices[i]['Close'], prices[i]['Volume']
            priceSet.append({ticker: prices})
            print num, ticker
        except:
            continue

    with open('./input/stockPrices.json', 'w') as outfile:
        json.dump(priceSet, outfile, indent=4)

def main():
    stock_Prices()

if __name__ == "__main__":
    main()