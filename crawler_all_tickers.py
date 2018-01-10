#!/usr/bin/env python3
from urllib.request import urlopen
import csv
import sys
import numpy as np

def get_tickers(percent):
    """Keep the top percent market-cap companies."""
    assert type(percent) is int

    file = open('./input/tickerList.csv', 'w')
    writer = csv.writer(file, delimiter=',')
    capStat, output = np.array([]), []
    for exchange in ["NASDAQ", "NYSE", "AMEX"]:        
        url = "http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange="
        repeat_times = 10 # repeat downloading in case of http error
        for _ in range(repeat_times): 
            try:
                print("Downloading tickers from {}...".format(exchange))
                response = urlopen(url + exchange + '&render=download')
                content = response.read().decode('utf-8').split('\n')
                for num, line in enumerate(content):
                    line = line.strip().strip('"').split('","')
                    if num == 0 or len(line) != 9: continue # filter unmatched format
                    ticker, name, lastSale, MarketCap, IPOyear, sector, \
                    industry = line[0: 4] + line[5: 8]
                    capStat = np.append(capStat, float(MarketCap))
                    output.append([ticker, name.replace(',', '').replace('.', ''), exchange, MarketCap])
                break
            except Exception as e:
                print(e)
                continue

    for data in output:
        marketCap = float(data[3])
        if marketCap < np.percentile(capStat, 100 - percent): continue
        writer.writerow(data)


def main():
    top_n = sys.argv[1]
    s = get_tickers(int(top_n)) # keep the top N% market-cap companies


if __name__ == "__main__":
    main()
