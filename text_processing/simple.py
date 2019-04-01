from pathlib import Path

import bs4

from text_processing.base import preprocess


def parse(path):
    with open(path, 'r') as f:
        markup = '\n'.join(f)
        soup = bs4.BeautifulSoup(markup)
        return preprocess(soup.text)


if __name__ == '__main__':
    for f in Path("../out/docs_simple2").rglob("*.html"):
        text = parse(f)
        print(text)
