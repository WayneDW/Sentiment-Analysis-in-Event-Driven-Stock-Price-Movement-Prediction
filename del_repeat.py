#!/usr/bin/env python3
"""
Roughly 600 companies have multiple tickers, they may have the same news, which can be confusing.
To deal with that, if a piece of news is published in two tickers, we only consider the primary one.

The logic is when a ticker is in the repeatedlist, but 
"""
import os
import sys
import json


def generate_list():
    dt = {}

    for l in open('./input/tickerList.csv'):
        l = l.strip().split(',')
        name = l[1]
        ticker = l[0]
        if name not in dt:
            dt[name] = []
        dt[name].append(ticker)

    cnt = 0

    filterlist = set()
    for name in dt:
        if len(dt[name]) > 1:
            for _ in range(1, len(dt[name])):
                filterlist.add(sorted(dt[name])[_])
            cnt += 1

    return(filterlist)


"""
Change news type from topstory to repeated
"""
def modify_news(date, filterlist):
    f = open('input/news/2018/news_' + date + '.csv')
    fout = open('input/news/2018/news_' + date + '.csv_bak', 'w')
    for l in f:
        l = l.strip().split(',')
        if len(l) == 6:
            ticker, company, timestamp, title, body, news_type = l
        elif len(l) == 7:
            ticker, company, timestamp, title, body, news_type, suggestion = l
        else:
            continue
        if news_type == 'topStory' and ticker in filterlist:
            print(news_type)
            news_type = 'repeated'
        if len(l) == 6:
            fout.write(','.join([ticker, company, timestamp, title, body, news_type])+ '\n')
        elif len(l) == 7:
            fout.write(','.join([ticker, company, timestamp, title, body, news_type, suggestion])+ '\n')
    fout.close()
    f.close()
    os.system('mv input/news/2018/news_' + date + '.csv_bak input/news/2018/news_' + date + '.csv')

    


def main():
    date = sys.argv[1]
    filterlist = generate_list()
    modify_news(date, filterlist)

if __name__ == '__main__':
    main()
