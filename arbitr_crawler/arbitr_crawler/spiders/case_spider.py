# -*- coding: utf-8 -*-
import json
import logging

import scrapy
from scrapy import FormRequest, Request


class CaseSpiderSpider(scrapy.Spider):
    name = 'case_spider'
    allowed_domains = ['kad.arbitr.ru']
    start_urls = ['http://kad.arbitr.ru/']

    def start_requests(self):
        def form_data(p):
            return json.dumps({
                'Page': p,
                'Count': 25,
                'CaseType': 'G',
                'Courts': [],
                'DateFrom': "2018-01-01T00:00:00",
                'DateTo': "2018-12-31T00:00:00",
                'WithVKSInstances': False
            })

        headers = {
            'Origin': 'http://kad.arbitr.ru',
            'Referer': 'http://kad.arbitr.ru/',
            'X-Requested-With': 'XMLHttpRequest',
            'Content-Type': 'application/json'
        }

        for i in range(1, 5):
            yield FormRequest('http://kad.arbitr.ru/Kad/SearchInstances',
                              headers=headers, body=form_data(i), callback=self.parse)

    def parse(self, response):
        for row in response.xpath('//tr'):
            url = row.xpath('//a[@class="num_case"]/@href').get()
            yield Request(url, callback=self.parse_case)

    def parse_case(self, response):
        case_num = response.xpath('//title/text()').get().strip()

        logging.info(response.xpath('//div[@id="gr_case_acts"]').get())
        for doc_row in response.xpath('//div[@id="gr_case_acts"]//tbody//tr'):
            logging.info(doc_row.xpath('//a/text()').get().strip())
            yield {
                'case_num': case_num,
                'case_url': response.url,
                'doc_name': doc_row.xpath('//a/text()').get().strip(),
                'doc_url': doc_row.xpath('//a/@href').get()
            }
