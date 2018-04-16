#!/usr/bin/python -u
import re
import urllib2
import csv
import os
import sys
import time
import datetime

import numpy as np
from bs4 import BeautifulSoup




# iterate over the past days
#   iterate all tickers in NYSE, NASDAQ and American Stock Exchange
#     repeatDowdload
#       save to ./input/data/news_date.csv

class news_Reuters:
    def __init__(self):
        fin = open('./input/tickerList.csv')
        fin = fin.readlines() # make sure every time we iterate fin, pointer is reset to the beginning

        filterList = set()
        try: # this is used when we restart a task
            fList = open('./input/finished.reuters')
            for l in fList:
                filterList.add(l.strip())
        except: pass

        # https://uk.reuters.com/info/disclaimer
        # e.g. http://www.reuters.com/finance/stocks/company-news/BIDU.O?date=09262017
        self.suffix = {'AMEX': '.A', 'NASDAQ': '.O', 'NYSE': '.N'}
        self.repeat_times = 4
        self.sleep_times = 2
        self.iterate_by_day(fin, filterList)


    def iterate_by_day(self, fin, filterList):
        dateList = self.dateGenerator('20161201', '20170301')
        for timestamp in dateList: # iterate all possible days
            print("%s%s%s" % (''.join(['-'] * 50), timestamp, ''.join(['-'] * 50)))
            self.iterate_by_ticker(fin, filterList, timestamp)

    def iterate_by_ticker(self, fin, filterList, timestamp):
        for line in fin: # iterate all possible tickers
            line = line.strip().split(',')
            ticker, name, exchange, MarketCap = line
            if ticker in filterList: continue
            print("%s - %s - %s - %s" % (ticker, name, exchange, MarketCap))
            self.repeatDownload(ticker, line, timestamp, exchange)

    def repeatDownload(self, ticker, line, timestamp, exchange): 
        url = "https://www.reuters.com/finance/stocks/company-news/" + ticker + self.suffix[exchange]
        new_time = timestamp[4:] + timestamp[:4] # change 20151231 to 12312015 to match reuters format
        for _ in range(self.repeat_times): 
            try:
                time.sleep(np.random.poisson(self.sleep_times))
                response = urllib2.urlopen(url + "?date=" + new_time)
                data = response.read()
                soup = BeautifulSoup(data, "lxml")
                hasNews = self.parser(soup, line, ticker, timestamp)
                if hasNews: return 1 # return if we get the news
                break # stop looping if the content is empty (no error)
            except: # repeat if http error appears
                print('Http error')
                continue
        return 0
  
    def parser(self, soup, line, ticker, timestamp):
        content = soup.find_all("div", {'class': ['topStory', 'feature']})
        if len(content) == 0: return 0
        fout = open('./input/dates/news_' + timestamp + '.csv', 'a+')
        for i in range(len(content)):
            title = content[i].h2.get_text().replace(",", " ").replace("\n", " ")
            body = content[i].p.get_text().replace(",", " ").replace("\n", " ")

            if i == 0 and len(soup.find_all("div", class_="topStory")) > 0: news_type = 'topStory'
            else: news_type = 'normal'

            print(ticker, timestamp, title, news_type)
            fout.write(','.join([ticker, line[1], timestamp, title, body, news_type]).encode('utf-8') + '\n')
        fout.close()
        return 1
    
    def dateGenerator(self, start, end): # generate weekdays between start and end, including boundary
        start = datetime.datetime.strptime(start, '%Y%m%d')
        end = datetime.datetime.strptime(end, '%Y%m%d')
        dlist = [end - datetime.timedelta(days=x) for x in range((end - start).days + 1)]
        days = [day.strftime("%Y%m%d") for day in dlist if day.weekday() not in [5, 6]]
        return days

def main():
    news_Reuters()

if __name__ == "__main__":
    main()
