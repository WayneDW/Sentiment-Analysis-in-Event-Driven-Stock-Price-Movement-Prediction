#-*- coding=utf-8 -*-
# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import os

class ScrapynewsPipeline(object):
    def process_item(self, item, spider):

    	item['news_content'] = ' '.join(item['news_content'])
    	item['news_authors'] = ' '.join(item['news_authors'])

    	if not os.path.isdir(item["news_domain"]):
    		os.mkdir(item["news_domain"])

    	if not os.path.isdir(item["news_domain"]+"/"+item["news_dir"]):
    		os.mkdir(item["news_domain"]+"/"+item["news_dir"])

    	with open(item["news_domain"]+"/"+item["news_dir"] + '/' + item['news_filename'],'w') as fp:
    		fp.write('-- ' + item['news_title'] + '\n')
    		fp.write('-- ' + item['news_authors'] + '\n')
    		fp.write('-- ' + item['news_posttime'] + '\n')
    		fp.write('-- ' + item['news_url'] + '\n')
    		fp.write( item['news_content'].encode('utf-8'))
