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
            break

    def content(self, ticker, line, dateList):
        # http://www.reuters.com/finance/stocks/companyNews?symbol=GOOGL.O&date=11162016
        url = "http://www.reuters.com/finance/stocks/companyNews?symbol=" + ticker 
        for timestamp in dateList:
            print(line, timestamp)
            tag = self.repeatDownload(ticker, line, url, timestamp) # if page has news, tag 1, otherwise tag 0
            if not tag: break
            #break

    def dateGenerator(self, numdays): # generate N days until now
        base = datetime.datetime.today()
        date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
        for i in range(len(date_list)): date_list[i] = date_list[i].strftime("%Y%m%d")
        return date_list


    def repeatDownload(self, ticker, line, url, timestamp): 
        new_time = timestamp[4:] + timestamp[:4] # change 20151231 to 12312015 to satisfy reuters format
        repeat_times = 4 # repeat downloading in case of http error
        tag = 0 
        for _ in range(repeat_times): 
            try:
                time.sleep(np.random.poisson(5))
                response = urllib2.urlopen(url + "&date=" + new_time)
                data = response.read()
                soup = BeautifulSoup(data, "lxml")
                iftag = self.parser(soup, line, ticker, timestamp)
                if iftag: tag = 0 # if we receive news from that day, reset tag as 0
                tag += iftag
                if tag > 30: return 0 # if a stock has no news for 30 consecutive days, skip this stock
                break # skip loop if data fetched
            except:
                continue
        return 1

            

    def parser(self, soup, line, ticker, timestamp):
        content = soup.find_all("div", class_="feature")
        fout = open('./input/news_reuters.csv', 'a+')
        
        if len(content) == 0: return 1
        for i in range(len(content)):    
            title = ' '.join(re.findall(r"\w+", content[i].a.get_text()))
            body = ' '.join(re.findall(r"\w+", content[i].p.get_text()))
            fout.write(','.join([ticker, line[1], timestamp, title, body]) + '\n')
        fout.close()
        return 0
    
    


def main():
    newsCrawler()
    


if __name__ == "__main__":
    main()
