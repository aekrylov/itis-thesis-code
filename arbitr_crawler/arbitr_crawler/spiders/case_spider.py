# -*- coding: utf-8 -*-
import datetime
import json
import re
import time

import scrapy
from scrapy import FormRequest


class CaseSpiderSpider(scrapy.Spider):
    name = 'case_spider'
    allowed_domains = ['kad.arbitr.ru']

    def start_requests(self):
        def form_data(p, start, end):
            return json.dumps({
                'Page': p,
                'Count': 25,
                'CaseType': 'G',
                'Courts': [],
                "DateFrom": start.strftime("%Y-%m-%dT%H:%M:%S"),
                "DateTo": end.strftime("%Y-%m-%dT%H:%M:%S"),
                'WithVKSInstances': False
            })

        headers = {
            'Origin': 'http://kad.arbitr.ru',
            'Referer': 'http://kad.arbitr.ru/',
            'X-Requested-With': 'XMLHttpRequest',
            'Content-Type': 'application/json'
        }

        chunks = 365

        start_date = datetime.datetime(2017, 1, 1)
        end_date = datetime.datetime(2018, 31, 12)

        delta = (end_date - start_date) / chunks

        for chunk in range(0, chunks):
            start = start_date + delta*chunk
            end = start + delta

            for i in range(1, 2):
                yield FormRequest('http://kad.arbitr.ru/Kad/SearchInstances', method='POST',
                                  headers=headers, body=form_data(i, start, end), callback=self.parse)

    def parse(self, response):
        for row in response.xpath('//tr'):
            url = row.xpath('.//a[@class="num_case"]/@href').get()
            case_num = row.xpath('.//a[@class="num_case"]/text()').get().strip()
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
            form_data = {
                '_': str(int(time.time())),
                'id': case_id
            }

            yield FormRequest('http://kad.arbitr.ru/Case/Finalacts', meta=meta, headers=headers, formdata=form_data,
                              method='GET', callback=self.parse_case_json)

    def parse_case_json(self, response):
        obj = json.loads(response.body)
        # TODO multiple documents
        try:
            doc = obj['Result'][0]['FinalDocuments'][0]
        except IndexError:
            return
        except KeyError:
            return

        result = {
            'case_id': response.meta['case_id'],
            'case_num': response.meta['case_num'],
            'doc_id': doc['Id'],
            'doc_name': doc['FileName'],
        }

        result['doc_url'] = 'http://kad.arbitr.ru/PdfDocument/%s/%s/%s' \
                            % (result['case_id'], result['doc_id'], result['doc_name'])
        yield result
