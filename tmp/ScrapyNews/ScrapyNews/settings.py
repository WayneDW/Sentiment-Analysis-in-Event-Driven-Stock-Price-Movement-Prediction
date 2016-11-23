# Scrapy settings for ScrapyNews project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#

BOT_NAME = 'ScrapyNews'

SPIDER_MODULES = ['ScrapyNews.spiders']
NEWSPIDER_MODULE = 'ScrapyNews.spiders'

ITEM_PIPELINES = [
    'ScrapyNews.pipelines.ScrapynewsPipeline',
]

#DOWNLOADER_MIDDLEWARES = {
#	'scrapy.contrib.downloadermiddleware.httpproxy.HttpProxyMiddleware': 110,
#	'ScrapyNews.ProxyMiddleware.ProxyMiddleware': 100,
#}

SCHEDULER_ORDER = 'DFO'
CONCURRENT_REQUESTS = 50

SCHEDULER_PERSIST = True


# Crawl responsibly by identifying yourself (and your website) on the user-agent
USER_AGENT = 'ScrapyNews'
DOWNLOAD_DELAY = 2
DOWNLOAD_TIMEOUT = 25
