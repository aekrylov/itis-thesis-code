# -*- coding: utf-8 -*-
import json
import logging
import re
import time

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
            case_num = row.xpath('//a[@class="num_case"]/text()').get().strip()
            case_id = re.search(r'/([a-z0-9\-]+)$', url).group(1)
            meta = {
                'case_num': case_num,
                'case_id': case_id
            }

            headers = {
                'Referer': 'http://kad.arbitr.ru/Card/%s' % case_id,
                'X-Requested-With': 'XMLHttpRequest',
                'Content-Type': 'application/json'
            }

            yield FormRequest('http://kad.arbitr.ru/Case/Finalacts', meta=meta, headers=headers,
                              method='GET', formdata={'_': str(int(time.time())), 'id': case_id}, callback=self.parse_case_json)
            # yield Request(url, callback=self.parse_case)

    def parse_case(self, response):
        case_num = response.xpath('//title/text()').get().strip()

        logging.info(response.xpath('//div[@id="gr_case_acts"]').get())
        # TODO run JS and click on the tab
        for doc_row in response.xpath('//div[@id="gr_case_acts"]//tbody//tr'):
            logging.info(doc_row.xpath('//a/text()').get().strip())
            yield {
                'case_num': case_num,
                'case_url': response.url,
                'doc_name': doc_row.xpath('//a/text()').get().strip(),
                'doc_url': doc_row.xpath('//a/@href').get()
            }

    def parse_case_json(self, response):
        obj = json.loads(response.body)
        # TODO multiple documents
        doc = obj['Result'][0]['FinalDocuments'][0]

        result = {
            'case_id': response.meta['case_id'],
            'case_num': response.meta['case_num'],
            'doc_id': doc['Id'],
            'doc_name': doc['FileName'],
        }

        result['doc_url'] = 'http://kad.arbitr.ru/PdfDocument/%s/%s/%s' \
                            % (result['case_id'], result['doc_id'], result['doc_name'])
        yield result
