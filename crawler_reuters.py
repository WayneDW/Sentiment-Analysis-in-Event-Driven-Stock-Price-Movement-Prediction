#!/usr/bin/python
import re
import urllib2
import csv
import os
import sys
import time
import datetime

import numpy as np
from bs4 import BeautifulSoup

'''
Other useful crawler can be found in 
https://doc.scrapy.org/en/latest/
github.com/qolina/ScrapyFinanceNews/tree/master/ScrapyNews/spiders
'''

class news_Reuters:
    def __init__(self):
        fin = open('./input/tickerList.csv')
        # exit if the output already existed
        #if os.path.isfile('./input/news_reuters_part2.csv'):
        #    sys.exit("Reuters news already existed!")

        filterList = set()
        try: # this is used when we restart a task
            fList = open('./input/finished.reuters')
            for l in fList:
                filterList.add(l.strip())
        except: pass

        dateList = self.dateGenerator(1000) # look back on the past 1000 days
        for line in fin:
            line = line.strip().split(',')
            ticker, name, exchange, MarketCap = line
            if ticker in filterList: continue
            self.content(ticker, name, line, dateList)

    def content(self, ticker, name, line, dateList):
        url = "http://www.reuters.com/finance/stocks/companyNews?symbol=" + ticker

        # some company even doesn't have a single news, stop bothering iterating dates if so
        has_Content = 0
        repeat_times = 4
        for _ in range(repeat_times): # in case of crawling failure
            try:
                time.sleep(np.random.poisson(3))
                response = urllib2.urlopen(url)
                data = response.read()
                soup = BeautifulSoup(data, "lxml")
                has_Content = len(soup.find_all("div", {'class': ['topStory', 'feature']}))
                break
            except:
                continue
        
        if has_Content > 0:
            missing_days = 0
            print(ticker, name)
            for timestamp in dateList:
                hasNews = self.repeatDownload(ticker, line, url, timestamp) 
                if hasNews: missing_days = 0 # if get news, reset missing_days as 0
                else: missing_days += 1
                if missing_days > has_Content * 5 + 20: # 2 NEWS: wait 30 days and stop, 10 news, wait 70 days
                    break # no news in X consecutive days, stop crawling
                if missing_days > 0 and missing_days % 20 == 0: # print the process
                    print(ticker, "has no news for ", missing_days, " days")
        else:
            print(ticker, "has no single news")

    def repeatDownload(self, ticker, line, url, timestamp): 
        new_time = timestamp[4:] + timestamp[:4] # change 20151231 to 12312015 to satisfy reuters format
        repeat_times = 3 # repeat downloading in case of http error
        for _ in range(repeat_times): 
            try:
                time.sleep(np.random.poisson(3))
                response = urllib2.urlopen(url + "&date=" + new_time)
                data = response.read()
                soup = BeautifulSoup(data, "lxml")
                hasNews = self.parser(soup, line, ticker, timestamp)
                if hasNews: return 1 # return if we get the news
                break # stop looping if the content is empty (no error)
            except: # repeat if http error appears
                continue
        return 0
  
    def parser(self, soup, line, ticker, timestamp):
        content = soup.find_all("div", {'class': ['topStory', 'feature']})
        if len(content) == 0: return 0
        fout = open('./input/news_reuters.csv', 'a+')
        for i in range(len(content)):
            title = content[i].h2.get_text().replace(",", " ").replace("\n", " ")
            body = content[i].p.get_text().replace(",", " ").replace("\n", " ")

            if i == 0 and len(soup.find_all("div", class_="topStory")) > 0: news_type = 'topStory'
            else: news_type = 'normal'

            print(ticker, timestamp, title, news_type)
            fout.write(','.join([ticker, line[1], timestamp, title, body, news_type]).encode('utf-8') + '\n')
        fout.close()
        return 1
    
    def dateGenerator(self, numdays): # generate N days until now
        base = datetime.datetime.today()
        date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
        for i in range(len(date_list)): date_list[i] = date_list[i].strftime("%Y%m%d")
        return date_list

def main():
    news_Reuters()

if __name__ == "__main__":
    main()
