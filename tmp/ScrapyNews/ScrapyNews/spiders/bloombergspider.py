#-*- coding=utf-8 -*-
from scrapy import Spider
from scrapy import Selector
from scrapy.http import Request
from datetime import date,timedelta
from ScrapyNews.items import NewsItem
import re 

class BloombergSpider(Spider):
    name = 'bloombergspider'
    allowed_domains = ['bloomberg.com']

    start_urls = [
        "http://www.bloomberg.com",
        "http://www.bloomberg.com/markets",
        "http://www.bloomberg.com/insights",
        "http://www.bloomberg.com/live",
    ]

    parsed_urls = []

    start_time = "2015-04-10" #date(2015,04,10)
    end_time = "2015-05-31" #date(2015,05,31)

    def __init__(self):
        super(BloombergSpider,self).__init__()
        self.bloomberg_news_url = "http://www.bloomberg.com/archive/news/"
        self.bloomberg_url = 'http://www.bloomberg.com/'
        self.date_pattern = re.compile('([0-9]{4}-[0-9]{2}-[0-9]{2})')


    def parse(self, response):
        ## article/h1/a  or article/a
        #//article[@data-tracker-name="story"]

        for href in response.xpath('//a[@data-resource-type="article"]/@href'):
            url = href.extract()
            if not url.startswith("http://"):
                url =  self.bloomberg_url + url

            # time filtering
#            post_date = self.date_pattern.search(url)
#            if post_date is not None:
#                curr_date = post_date.group(1)
#                if curr_date < self.start_time:
#                    continue
#                if curr_date > self.end_time:
#                    continue

#            print url

            yield Request(url, callback=self.parse_news)

        for href in response.xpath('//a[@class="navigation-submenu__category-link"]/@href'):
            url = href.extract()
            if not url.startswith("http://"):
                url =  self.bloomberg_url + url

            if url in self.parsed_urls:
                continue
            self.parsed_urls.append(url)

            yield Request(url, callback=self.parse)



    def parse_news(self,response):
        debug = False

        if response.url in self.parsed_urls:
            return
        self.parsed_urls.append(response.url)
        
        if debug:
            print "********************************************************************"
            print response.url

        newsItem = NewsItem()
        hxs = Selector(response)
        try:
            newsItem['news_title'] = hxs.xpath('//h1[@class="lede-headline"]/span/text()').extract()[0].encode("utf-8")
            article_info = hxs.xpath('//div[@class="article-details"]')

            # arr-format
            newsItem['news_authors'] = article_info.xpath('//a[@class="author-link"]//text()').extract()

            newsItem['news_posttime'] = article_info.xpath('//div[@class="published-info"]/time/text()').extract()[0].encode("utf-8")

            # arr-format
            newsItem['news_content'] = hxs.xpath('//div[@class="article-body__content"]/p//text()').extract()


            newsItem['news_url'] = response.url
            newsItem['news_filename'] = response.url.rsplit('/',1)[1].strip(".html")

            post_date = self.date_pattern.search(response.url)
            
            if post_date is not None:
                newsItem['news_dir'] = post_date.group(1)

            newsItem['news_domain'] = "bloomberg"

            if debug:
                print "title", newsItem['news_title']
                print "authors", newsItem['news_authors']
                print "posttime", newsItem['news_posttime']
                print "content", "".join(newsItem['news_content'])[:20]
                print "dir", newsItem["news_dir"]
                print "filename", newsItem["news_filename"]

            yield newsItem
        except Exception,e:
            print '[ERROR]',e
