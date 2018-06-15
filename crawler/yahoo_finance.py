#!/usr/bin/env python3
"""
Crawl daily price data from yahoo finance to generate raw data

Require "./input/news_reuters.csv"
==> "./input/finished.reuters" by calc_finished_ticker()
==> "./input/stockPrices_raw.json" by get get_stock_prices()
json structure:
         ticker
        /  |   \
    open close adjust ...
      /    |     \
   dates dates  dates ...
"""
import sys
import re
import os
import time
import random
import json

# Credit: https://github.com/c0redumb/yahoo_quote_download/blob/master/yahoo_quote_download/yqd.py
from yqd import load_yahoo_quote


def calc_finished_ticker():
    os.system("awk -F',' '{print $1}' ./input/news_reuters.csv | sort | uniq > ./input/finished.reuters")


def get_stock_prices():
    fin = open('./input/finished.reuters')
    output = './input/stockPrices_raw.json'

    # exit if the output already existed
    if os.path.isfile(output):
        sys.exit("Prices data already existed!")

    price_set = {}
    price_set['^GSPC'] = repeat_download('^GSPC') # download S&P 500
    for num, line in enumerate(fin):
        ticker = line.strip()
        print(num, ticker)
        price_set[ticker] = repeat_download(ticker)
        # if num >= 10: break # for testing purpose

    with open(output, 'w') as outfile:
        json.dump(price_set, outfile, indent=4)


def repeat_download(ticker, start_date='20040101', end_date='29991201'):
    repeat_times = 3 # repeat download for N times
    for i in range(repeat_times):
        try:
            time.sleep(random.uniform(3, 5))
            price_str = get_price_from_yahoo(ticker, start_date, end_date)
            if price_str: # skip loop if data is not empty
                return price_str
        except Exception as e:
            print(e)
            if i == 0:
                print(ticker, "Http error!")

def get_price_from_yahoo(ticker, start_date, end_date):
    quote = load_yahoo_quote(ticker, start_date, end_date)

    # get historical price
    ticker_price = {}
    index = ['open', 'high', 'low', 'close', 'adjClose', 'volume']
    for num, line in enumerate(quote):
        line = line.strip().split(',')
        if len(line) < 7 or num == 0:
            continue
        date = line[0]
        # check if the date type matched with the standard type
        if not re.search(r'^[12]\d{3}-[01]\d-[0123]\d$', date):
            continue
        # open, high, low, close, volume, adjClose : 1,2,3,4,5,6
        for num, type_name in enumerate(index, 1):
            try:
                ticker_price[type_name][date] = round(float(line[num]), 2)
            except:
                ticker_price[type_name] = {}
    return ticker_price

if __name__ == "__main__":
    calc_finished_ticker()
    get_stock_prices()
