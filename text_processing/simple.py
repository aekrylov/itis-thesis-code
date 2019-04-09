import re
from pathlib import Path

import bs4

from text_processing.base import preprocess


def parse(path):
    with open(path, 'r') as f:
        markup = '\n'.join(f)
        soup = bs4.BeautifulSoup(markup, "lxml")
        return preprocess(soup.text)


def parse_all(basedir):
    for f in Path(basedir).rglob("*.html"):
        with open(f, 'r') as fd:
            markup = '\n'.join(fd)
            soup = bs4.BeautifulSoup(markup, "lxml")
            if not re.search(r'установил\s*:\s*\n', soup.text, re.IGNORECASE | re.MULTILINE):
                continue
            if re.search(r'Судебный акт принят в закрытом судебном заседании', soup.text):
                continue
            yield preprocess(soup.text)


if __name__ == '__main__':
    for doc in parse_all("../out/docs_simple2"):
        print(doc)
