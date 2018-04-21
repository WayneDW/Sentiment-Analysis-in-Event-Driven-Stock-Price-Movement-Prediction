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
    # if os.path.isfile('./input/stockPrices.json'):
    #     sys.exit("Prices data already existed!")

    priceSet = {}
    fin = open('./input/tickerList.csv')
    for num, line in enumerate(fin):
        line = line.strip().split(',')
        ticker, name, exchange, MarketCap = line
        if 1:
            print(num, ticker)
            yahoo = Share(ticker)
            time.sleep(np.random.poisson(3))
            prices = yahoo.get_historical('2005-01-01', '2020-01-01')
            priceDt = {}
            for i in range(len(prices)):
                date = ''.join(prices[i]['Date'].split('-'))
                priceDt[date] = round(log(float(prices[i]['Close']) / float(prices[i]['Open'])), 6)
            priceSet[ticker] = priceDt
        #except:
            #continue

    with open('./input/stockPrices.json', 'w') as outfile:
        json.dump(priceSet, outfile, indent=4)

def main():
    stock_Prices()

if __name__ == "__main__":
    main()
