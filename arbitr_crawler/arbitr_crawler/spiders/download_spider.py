import json
import os

import scrapy
from scrapy import Request


class DownloadSpider(scrapy.Spider):
    name = 'download_spider'
    allowed_domains = ['kad.arbitr.ru']

    def _file_path(self, doc):
        return '../out/docs/%s_%s' % (doc['case_id'], doc['doc_name'])

    def start_requests(self):
        with open('../out/docs.json', 'r') as f:
            for line in f.readlines():
                doc = json.loads(line)
                if not os.path.isfile(self._file_path(doc)):
                    yield Request(doc['doc_url'], meta=doc, headers={'User-Agent': 'Wget/1.19.4 (linux-gnu)'})

    def parse(self, response):
        with open(self._file_path(response.meta), 'wb') as f:
            f.write(response.body)
