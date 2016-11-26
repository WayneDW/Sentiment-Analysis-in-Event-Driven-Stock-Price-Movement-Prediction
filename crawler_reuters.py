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
        if os.path.isfile('./input/news_reuters.csv'):
            sys.exit("Reuters news already existed!")

        dateList = self.dateGenerator(1000)
        for line in fin:
            line = line.strip().split(',')
            ticker, name, exchange, MarketCap = line
            self.content(ticker, line, dateList)

    def content(self, ticker, line, dateList):
        # http://www.reuters.com/finance/stocks/companyNews?symbol=GOOGL.O&date=11162016
        url = "http://www.reuters.com/finance/stocks/companyNews?symbol=" + ticker 
        tag = 0
        for timestamp in dateList:
            print(line, timestamp)
            hasNews = self.repeatDownload(ticker, line, url, timestamp) 
            if hasNews: tag = 0 # if get news, reset tag as 0
            else: tag += 1
            if tag > 30: break # if we can't receive news in 30 consecutive days, stop this ticker

    def repeatDownload(self, ticker, line, url, timestamp): 
        new_time = timestamp[4:] + timestamp[:4] # change 20151231 to 12312015 to satisfy reuters format
        repeat_times = 4 # repeat downloading in case of http error
        for _ in range(repeat_times): 
            try:
                time.sleep(np.random.poisson(5))
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
        content = soup.find_all("div", class_="feature")
        fout = open('./input/news_reuters.csv', 'a+')
        
        if len(content) == 0: return 0
        for i in range(len(content)):    
            title = ' '.join(re.findall(r"\w+", content[i].a.get_text()))
            body = ' '.join(re.findall(r"\w+", content[i].p.get_text()))
            fout.write(','.join([ticker, line[1], timestamp, title, body]) + '\n')
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
