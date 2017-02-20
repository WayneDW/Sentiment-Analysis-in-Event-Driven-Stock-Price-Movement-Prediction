#!/usr/bin/python
import re
import urllib2
import csv
import os
import sys
import time

import numpy as np
from bs4 import BeautifulSoup

'''
Other useful crawler can be found in 

How to install scrapy
https://doc.scrapy.org/en/latest/intro/install.html
http://askubuntu.com/questions/675876/cant-install-some-packages-depends-x-but-x-is-to-be-installed
'''

class news_Bloomberg:
    def __init__(self):
        fin = open('./input/tickerList.csv')
        # exit if the output already existed
        if os.path.isfile('./input/news_bloomberg.csv'):
            sys.exit("Bloomberg news already existed!")
        
        filterList = set()
        try:
            fList = open('./input/finished.list')
            for l in fList:
                filterList.add(l.strip())
        except: pass
        
        for line in fin:
            line = line.strip().split(',')
            ticker, name, exchange, MarketCap = line
            if ticker in filterList: continue
            print ticker
            self.content(ticker, line)

    def content(self, ticker, line):
        url = "http://www.bloomberg.com/search?sort=time:desc&query=" + ticker 
        for pn in range(1, 500):
            print(line, pn)
            tag = self.repeatDownload(ticker, line, url, pn) # if page has news, tag 1, otherwise tag 0
            if not tag: break

    def repeatDownload(self, ticker, line, url, pn): 
        repeat_times = 4 # repeat downloading in case of http error
        for _ in range(repeat_times): 
            try:
                time.sleep(np.random.poisson(3))
                response = urllib2.urlopen(url + "&page=" + str(pn))
                data = response.read()
                soup = BeautifulSoup(data, "lxml")
                tag = self.parser(soup, line, ticker)
                if not tag: return tag # if page has no news, return 0 to change ticker
                break # skip loop if data fetched
            except:
                continue
        return 1

    def parser(self, soup, line, ticker):
        timeSet = soup.find_all("div", class_="search-result-story__metadata")
        titles = soup.find_all("h1", class_="search-result-story__headline")
        tags = soup.find_all("div", class_="search-result-story__body")
        fout = open('./input/news_bloomberg.csv', 'a+')
        if len(timeSet) == 0: return 0
        for i in range(len(timeSet)):    
            timestamp = self.timeConvert(timeSet[i].time.get_text())
            title = " ".join(re.findall(r"\w+", titles[i].a.get_text()))
            abstract = " ".join(re.findall(r"\w+", tags[i].get_text()))
            fout.write(','.join([ticker, line[1],timestamp, title, abstract]) + '\n')
        fout.close()
        return 1

    def timeConvert(self, time):
        conv = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', \
                'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08', \
                'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
        timeFmt = ' '.join(re.findall(r"\w+", time)) # format Nov 22 2016
        periods = timeFmt.split(" ")
        if len(periods[1]) == 1: periods[1] = '0' + periods[1]
        newTime = ''.join([periods[2], conv[periods[0]], periods[1]]) # format 20161122
        return newTime
    
    


def main():
    news_Bloomberg()
    


if __name__ == "__main__":
    main()
