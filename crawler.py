import re
import urllib2
import csv
import random
import time


from sets import Set
from bs4 import BeautifulSoup

'''
Other useful crawler can be found in 
https://doc.scrapy.org/en/latest/
github.com/qolina/ScrapyFinanceNews/tree/master/ScrapyNews/spiders

'''

class newsCrawler:
    def __init__(self):
        fin = open('./input/tickerList.csv')
        fout = open('./input/bloombergNews.txt', 'a+')
        writer = csv.writer(fout, delimiter=',')
        for line in fin:
            line = line.strip().split(',')
            ticker, name, exchange, MarketCap = line
            self.content(ticker, writer, line)
        fout.close()

    def content(self, ticker, writer, line):
        url = "http://www.bloomberg.com/search?sort=time:desc&query=" + ticker 
        for pn in range(1, 1000):
            print(line, pn)
            time.sleep(random.uniform(3, 10))
            response = urllib2.urlopen(url + "&page=" + str(pn))
            csv = response.read()
            soup = BeautifulSoup(csv, "lxml")
            self.parser(soup, ticker, writer)
            

    def parser(self, soup, ticker, writer):
        timeSet = soup.find_all("div", class_="search-result-story__metadata")
        titles = soup.find_all("h1", class_="search-result-story__headline")
        tags = soup.find_all("div", class_="search-result-story__body")
        
        for i in range(len(timeSet)):
            timestamp = self.timeConvert(timeSet[i].time.get_text())
            title = " ".join(re.findall(r"\w+", titles[i].a.get_text()))
            tag = " ".join(re.findall(r"\w+", tags[i].get_text()))
            writer.writerow([ticker, timestamp, title, tag])

    def timeConvert(self, time):
        conv = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', \
                'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08', \
                'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
        timeFmt = ' '.join(re.findall(r"\w+", time)) # format Nov 22 2016
        periods = timeFmt.split(" ")
        newTime = ''.join([periods[2], conv[periods[0]], periods[1]]) # format 20161122
        return newTime
    
    


def main():
    newsCrawler()
    


if __name__ == "__main__":
    main()