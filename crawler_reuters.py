#!/usr/bin/env python3
import time
import datetime
from urllib.request import urlopen

import numpy as np
from bs4 import BeautifulSoup


class ReutersCrawler(object):
    def __init__(self):
        self.ticker_list = open('./input/tickerList.csv')

        self.finished_tickers = set()
        try: # this is used when we restart a task
            finished = open('./input/finished.reuters')
            for ticker in finished:
                self.finished_tickers.add(ticker.strip())
        except:
            pass

    def fetch_content(self, ticker, name, line, date_list, exchange):
        # https://uk.reuters.com/info/disclaimer
        suffix = {'AMEX': '.A', 'NASDAQ': '.O', 'NYSE': '.N'}
        # e.g. http://www.reuters.com/finance/stocks/company-news/BIDU.O?date=09262017
        url = "http://www.reuters.com/finance/stocks/companyNews/" + ticker + suffix[exchange]
        has_content = 0
        repeat_times = 4
        # check the website to see if that ticker has many news
        # if true, iterate url with date, otherwise stop
        for _ in range(repeat_times): # repeat in case of http failure
            try:
                time.sleep(np.random.poisson(3))
                response = urlopen(url)
                data = response.read().decode('utf-8')
                soup = BeautifulSoup(data, "lxml")
                has_content = len(soup.find_all("div", {'class': ['topStory', 'feature']}))
                break
            except:
                continue

        # spider task for the past
        # if some company has no news even if we don't input date
        #     set this ticker into the lowest priority list
        # else
        #     if it doesn't have a single news for NN consecutive days, stop iterating dates
        #     set this ticker into the second-lowest priority list
        ticker_failed = open('./input/news_failed_tickers.csv', 'a+')
        if has_content > 0:
            missing_days = 0
            for timestamp in date_list:
                has_news = self.repeat_download(ticker, line, url, timestamp)
                if has_news:
                    missing_days = 0 # if get news, reset missing_days as 0
                else:
                    missing_days += 1
                # 2 NEWS: wait 30 days and stop, 10 news, wait 70 days
                if missing_days > has_content * 5 + 20:
                    break # no news in X consecutive days, stop crawling
                if missing_days > 0 and missing_days % 20 == 0:
                    print("%s has no news for %d days, stop this candidate ..." % (ticker, missing_days))
                    ticker_failed.write(ticker + ',' + timestamp + ',' + 'LOW\n')
        else:
            print("%s has no news" % (ticker))
            today = datetime.datetime.today().strftime("%Y%m%d")
            ticker_failed.write(ticker + ',' + today + ',' + 'LOWEST\n')
        ticker_failed.close()

    def repeat_download(self, ticker, line, url, timestamp):
        new_time = timestamp[4:] + timestamp[:4] # change 20151231 to 12312015 to match reuters format
        repeat_times = 3 # repeat downloading in case of http error
        for _ in range(repeat_times):
            try:
                time.sleep(np.random.poisson(3))
                response = urlopen(url + "?date=" + new_time)
                data = response.read().decode('utf-8')
                soup = BeautifulSoup(data, "lxml")
                has_news = self.parser(soup, line, ticker, timestamp)
                if has_news:
                    return 1 # return if we get the news
                break # stop looping if the content is empty (no error)
            except: # repeat if http error appears
                print('Http error')
                continue
        return 0

    def parser(self, soup, line, ticker, timestamp):
        content = soup.find_all("div", {'class': ['topStory', 'feature']})
        if not content:
            return 0
        fout = open('./input/news_reuters.csv', 'a+')
        for i in range(len(content)):
            title = content[i].h2.get_text().replace(",", " ").replace("\n", " ")
            body = content[i].p.get_text().replace(",", " ").replace("\n", " ")

            if i == 0 and soup.find_all("div", class_="topStory"):
                news_type = 'topStory'
            else:
                news_type = 'normal'

            print(ticker, timestamp, title, news_type)
            fout.write(','.join([ticker, line[1], timestamp, title, body, news_type]).encode('utf-8') + '\n')
        fout.close()
        return 1

    def generate_past_n_day(self, numdays):
        """Generate N days until now"""
        base = datetime.datetime.today()
        date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
        return [x.strftime("%Y%m%d") for x in date_list]

    def run(self, numdays=1000):
        """Start crawler back to numdays"""
        date_list = self.generate_past_n_day(numdays) # look back on the past X days
        for line in self.ticker_list: # iterate all possible tickers
            line = line.strip().split(',')
            ticker, name, exchange, market_cap = line
            if ticker in self.finished_tickers:
                continue
            print("%s - %s - %s - %s" % (ticker, name, exchange, market_cap))
            self.fetch_content(ticker, name, line, date_list, exchange)


def main():
    reuter_crawler = ReutersCrawler()
    reuter_crawler.run(1)

if __name__ == "__main__":
    main()
