# -*- coding: utf-8 -*-
import scrapy


class CaseSpiderSpider(scrapy.Spider):
    name = 'case_spider'
    allowed_domains = ['kad.arbitr.ru']
    start_urls = ['http://kad.arbitr.ru/']

    def parse(self, response):
        pass
