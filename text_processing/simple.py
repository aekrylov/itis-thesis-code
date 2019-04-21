import os
import re
from pathlib import Path

import bs4
from joblib import Memory

from text_processing.base import preprocess


def cache_path(path):
    return str(path)\
        .replace('docs_simple2', 'docs_simple2_processed')\
        .replace('.html', '.txt')


def parse(path, from_cache=False):
    if from_cache:
        if os.path.exists(cache_path(path)):
            with open(cache_path(path), 'r') as fd:
                return '\n'.join(fd)
        else:
            return None

    with open(path, 'r') as fd:
        markup = '\n'.join(fd)
        soup = bs4.BeautifulSoup(markup, "lxml")
        if not re.search(r'установил\s*:\s*\n', soup.text, re.IGNORECASE | re.MULTILINE):
            return None
        if re.search(r'Судебный акт принят в закрытом судебном заседании', soup.text):
            return None
        return preprocess(soup.text)


def parse_all(basedir, mem: Memory = None, with_paths=False, from_cache=False):
    _parse = parse if mem is None else mem.cache(parse, ignore=['from_cache'])

    for f in Path(basedir).rglob("*.html"):
        text = _parse(f, from_cache)
        if text is not None:
            yield (f, text) if with_paths else text


if __name__ == '__main__':
    for path, doc in parse_all("../out/docs_simple2", with_paths=True):
        path = cache_path(path)
        if os.path.exists(path):
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as out:
            out.write(doc)
