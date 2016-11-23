# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

from scrapy.item import Item, Field

class NewsItem(Item):
	news_title = Field()
	news_authors = Field()
	news_posttime = Field()
	news_content = Field()
	news_url = Field()
	news_dir = Field()
	news_filename = Field()

        news_domain = Field()
