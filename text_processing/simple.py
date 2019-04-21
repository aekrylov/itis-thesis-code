import re
from pathlib import Path

import bs4

from text_processing.base import preprocess


def parse(path):
    with open(path, 'r') as fd:
        markup = '\n'.join(fd)
        soup = bs4.BeautifulSoup(markup, "lxml")
        if not re.search(r'установил\s*:\s*\n', soup.text, re.IGNORECASE | re.MULTILINE):
            return None
        if re.search(r'Судебный акт принят в закрытом судебном заседании', soup.text):
            return None
        return preprocess(soup.text)


def parse_all(basedir, mem=None):
    _parse = parse if mem is None else mem.cache(parse)

    for f in Path(basedir).rglob("*.html"):
        text = _parse(f)
        if text is not None:
            yield text


if __name__ == '__main__':
    for doc in parse_all("../out/docs_simple2"):
        print(doc)
