import urllib2
import csv
import numpy as np

def getTickers(percent):
    file = open('./input/tickerList.csv', 'w')
    writer = csv.writer(file, delimiter=',')
    capStat, output = np.array([]), []
    for exchange in ["NASDAQ", "NYSE", "AMEX"]:
        print "Download tickers from " + exchange
        u0 = "http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange="
        u1 = "&render=download"
        response = urllib2.urlopen(u0 + exchange + u1)
        content = response.read().split('\n')
        for num, line in enumerate(content):
            line = line.strip().strip('"').split('","')
            if num == 0 or len(line) != 9: continue # filter unmatched format
            ticker, name, lastSale, MarketCap, IPOyear, sector, \
            industry = line[0: 4] + line[5: 8]
            capStat = np.append(capStat, float(MarketCap))
            output.append([ticker, name.replace(',', '').replace('.', ''), exchange, MarketCap])

    for data in output:
        marketCap = float(data[3])
        if marketCap < np.percentile(capStat, 100 - percent): continue
        writer.writerow(data)


def main():
    s = getTickers(1) # keep the top N% market-cap companies


if __name__ == "__main__":
    main()