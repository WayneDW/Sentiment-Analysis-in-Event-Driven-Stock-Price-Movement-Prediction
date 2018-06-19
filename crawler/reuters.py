#!/usr/bin/env python3
"""
Crawl news headlines from Reuters and save as csv.
Input file: ./input/tickerList.csv
News append to: ./input/news_reuters.csv
"""
import os
import sys
import time
import datetime

# import util from parent directory
# credit: https://stackoverflow.com/a/11158224/4246348
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util


class ReutersCrawler(object):

    def __init__(self):
        self.ticker_list_filename = './input/tickerList.csv'
        self.finished_reuters_filename = './input/finished.reuters'
        self.failed_reuters_filename = './input/news_failed_tickers.csv'
        self.news_filename = './input/news_reuters.csv'

    def load_finished_tickers(self):
        # when we restart a task, we may call calc_finished_ticker() in crawler/yahoo_finance.py
        # so that we can load the already finished reuters if any
        return set(self._load_from_file(self.finished_reuters_filename))

    def load_failed_tickers(self):
        failed_tickers = {}  # {ticker: priority}
        for line in self._load_from_file(self.failed_reuters_filename):
            ticker, _, priority = line.split(',')
            failed_tickers[ticker] = priority
        return failed_tickers

    def _load_from_file(self, filename):
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                for line in f:
                    yield line.strip()

    def fetch_content(self, task, date_range):
        # https://uk.reuters.com/info/disclaimer
        ticker, name, exchange, market_cap = task
        print("%s - %s - %s - %s" % (ticker, name, exchange, market_cap))

        suffix = {'AMEX': '.A', 'NASDAQ': '.O', 'NYSE': '.N'}
        # e.g. https://www.reuters.com/finance/stocks/company-news/BIDU.O?date=09262017
        url = "https://www.reuters.com/finance/stocks/company-news/" + ticker + suffix[exchange]
        
        ticker_failed = open(self.failed_reuters_filename, 'a+')
        today = datetime.datetime.today().strftime("%Y%m%d")

        news_num = self.get_news_num_whenever(url)
        if news_num:
            # this company has news, then fetch for N consecutive days in the past
            has_content, no_news_days = self.fetch_within_date_range(news_num, url, date_range, task, ticker)
            if not has_content:
                print('%s has no content within date range' % ticker)
            if no_news_days:
                print('set as LOW priority')
                for timestamp in no_news_days:
                    ticker_failed.write(ticker + ',' + timestamp + ',' + 'LOW\n')
        else:
            # this company has no news even if we don't set a date
            # add it into the lowest priority list
            print("%s has no news at all, set as LOWEST priority" % (ticker))
            ticker_failed.write(ticker + ',' + today + ',' + 'LOWEST\n')
        ticker_failed.close()

    def get_news_num_whenever(self, url):
        # check the website to see if the ticker has any news
        # return the number of news
        soup = util.get_soup_with_repeat(url, repeat_times=4)
        if soup:
            return len(soup.find_all("div", {'class': ['topStory', 'feature']}))
        return 0

    def fetch_within_date_range(self, news_num, url, date_range, task, ticker):
        # if it doesn't have a single news for X consecutive days, stop iterating dates
        # set this ticker into the second-lowest priority list
        missing_days = 0
        has_content = False
        no_news_days = []
        for timestamp in date_range:
            print('trying '+timestamp, end='\r', flush=True)  # print timestamp on the same line
            new_time = timestamp[4:] + timestamp[:4] # change 20151231 to 12312015 to match reuters format
            soup = util.get_soup_with_repeat(url + "?date=" + new_time)
            if soup and self.parse_and_save_news(soup, task, ticker, timestamp):
                missing_days = 0 # if get news, reset missing_days as 0
                has_content = True
            else:
                missing_days += 1

            # the more news_num, the longer we can wait
            # e.g., if news_num is 2, we can wait up to 30 days; 10 news, wait up to 70 days
            if missing_days > news_num * 5 + 20:
                # no news in X consecutive days, stop crawling
                print("%s has no news for %d days, stop this candidate ..." % (ticker, missing_days))
                break
            if missing_days > 0 and missing_days % 20 == 0:
                no_news_days.append(timestamp)

        return has_content, no_news_days

    def parse_and_save_news(self, soup, task, ticker, timestamp):
        content = soup.find_all("div", {'class': ['topStory', 'feature']})
        if not content:
            return False
        with open(self.news_filename, 'a+', newline='\n') as fout:
            for i in range(len(content)):
                title = content[i].h2.get_text().replace(",", " ").replace("\n", " ")
                body = content[i].p.get_text().replace(",", " ").replace("\n", " ")

                if i == 0 and soup.find_all("div", class_="topStory"):
                    news_type = 'topStory'
                else:
                    news_type = 'normal'

                print(ticker, timestamp, title, news_type)
                # fout.write(','.join([ticker, task[1], timestamp, title, body, news_type]).encode('utf-8') + '\n')
                fout.write(','.join([ticker, task[1], timestamp, title, body, news_type])+ '\n')
        return True

    def run(self, numdays=1000):
        """Start crawler back to numdays"""
        finished_tickers = self.load_finished_tickers()
        failed_tickers = self.load_failed_tickers()
        date_range = util.generate_past_n_days(numdays) # look back on the past X days

        # store low-priority task and run later
        delayed_tasks = {'LOWEST': set(), 'LOW': set()}
        with open(self.ticker_list_filename) as ticker_list:
            for line in ticker_list:  # iterate all possible tickers
                task = tuple(line.strip().split(','))
                ticker, name, exchange, market_cap = task
                if ticker in finished_tickers:
                    continue
                if ticker in failed_tickers:
                    priority = failed_tickers[ticker]
                    delayed_tasks[priority].add(task)
                    continue
                self.fetch_content(task, date_range)

        # run task with low priority
        for task in delayed_tasks['LOW']:
            self.fetch_content(task, date_range)
        # run task with lowest priority
        for task in delayed_tasks['LOWEST']:
            self.fetch_content(task, date_range)


def main():
    reuter_crawler = ReutersCrawler()
    reuter_crawler.run(1)

if __name__ == "__main__":
    main()
