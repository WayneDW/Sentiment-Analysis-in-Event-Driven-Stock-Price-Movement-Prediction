import scrapy
from spider.items import newsItem


class QuotesSpider(scrapy.Spider):
    name = "bloomberg"
    start_urls = []
    for l in open('url_bloomberg'):
        start_urls.append(l.strip())

    def parse(self, response):
        for quote in response.css('div.container'):
            time = quote.css('span.metadata-timestamp time::text').extract()
            headline = quote.css('h1.search-result-story__headline a::text').extract()
            tag = quote.css('div.search-result-story__body::text').extract()
            for i in range(len(time)):
                output = newsItem()
                output['time'] = time[i]
                output['headline'] = headline[i]
                #output['tag'] = tag[i]
                yield output