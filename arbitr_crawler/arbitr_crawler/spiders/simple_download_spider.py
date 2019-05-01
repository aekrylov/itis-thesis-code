import json
import os
import random

import scrapy
from scrapy import FormRequest

from text_processing.simple import parse, cache_path


def get_path(doc):
    return '../out/docs_simple2/%s/%s.html' % (doc['case_id'][:2], doc['case_id'])


class DownloadSpider(scrapy.Spider):
    name = 'simple_download_spider'
    allowed_domains = ['ras.arbitr.ru']

    def start_requests(self):
        with open('../out/docs_simple4.json', 'r') as f:
            for line in f.readlines():
                doc = json.loads(line)
                if not os.path.isfile(get_path(doc)):
                    if random.random() > 0.07:
                        continue

                    yield FormRequest('http://ras.arbitr.ru/Ras/HtmlDocument/%s' % doc['doc_id'],
                                      formdata={'hilightText': 'null'},
                                      meta=doc, headers={'User-Agent': 'Wget/1.19.4 (linux-gnu)'})

    def parse(self, response):
        path = get_path(response.meta)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(response.body.decode('utf-8'))

        parsed = parse(path)
        if parsed is not None:
            with open(cache_path(path), 'w') as cache:
                cache.write(parsed)
