# -*- coding: utf-8 -*-
import datetime
import json
import logging
import os
import random

import scrapy
from scrapy import FormRequest

from arbitr_crawler.spiders.simple_download_spider import get_path


class SimpleCaseSpider(scrapy.Spider):
    name = 'simple_case_spider'
    allowed_domains = ['ras.arbitr.ru']

    search_headers = {
        'Origin': 'http://ras.arbitr.ru',
        'Referer': 'http://ras.arbitr.ru/',
        'X-Requested-With': 'XMLHttpRequest',
        'Content-Type': 'application/json'
    }

    skip_types = {
        'e6dd1e2a-d64e-44bb-8ef2-614d53e432a1',  # Судебный приказ
        'c922ae18-151f-4fda-93d7-b442d4555a06',  # Резолютивная часть решения суда по делу, рассматриваемому в порядке упрощенного производства
    }

    def _form_data(self, p, start, end):
        return {
            "GroupByCase": False,
            "Count": 25,
            "Page": p,
            "DisputeTypes": ["1782f653-0cbb-44b3-beab-067d6fa57c20"],
            "DateFrom": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "DateTo": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "Sides": [],
            "Judges": [],
            "Cases": [],
            "Text": "",
            "InstanceType": ["1"],
            "IsFinished": "1"
        }

    def start_requests(self):
        chunks = 365

        start_date = datetime.datetime(2018, 1, 1)
        end_date = datetime.datetime(2018, 12, 31)

        delta = (end_date - start_date) / chunks

        for i in range(chunks):
            start = start_date + delta*i
            end = start + delta

            form_data = self._form_data(1, start, end)

            yield FormRequest('http://ras.arbitr.ru/Ras/Search', method='POST', meta=form_data,
                              headers=self.search_headers, body=json.dumps(form_data), callback=self.parse_first_page)

    def parse_first_page(self, response):
        result = json.loads(response.body)
        if not result['Success']:
            logging.warning(result['Message'])
            return

        # parse current page
        self.parse(response)

        # query other pages
        pages = result['Result']['PagesCount']
        pages_sample = list(range(2, pages+1))
        random.shuffle(pages_sample)

        # only use random subset of pages
        form_data = response.meta

        for i in pages_sample:
            form_data["Page"] = i
            yield FormRequest('http://ras.arbitr.ru/Ras/Search', method='POST',
                              headers=self.search_headers, body=json.dumps(form_data), callback=self.parse)

    def parse(self, response):
        result = json.loads(response.body)
        if not result['Success']:
            logging.warning(result['Message'])
            return

        items = result['Result']['Items']
        for item in items:
            result = {
                'case_id': item['CaseId'],
                'case_num': item['CaseNumber'],
                'doc_id': item['Id'],
                'doc_name': item['FileName'],
            }

            if any(skip_type in item['ContentTypesString'] for skip_type in self.skip_types):
                try:
                    if os.path.exists(get_path(result)):
                        logging.info('Removing %s, content types[0]: %s' % (get_path(result), item['ContentTypes'][0]))
                except FileNotFoundError:
                    pass
                continue

            yield result
