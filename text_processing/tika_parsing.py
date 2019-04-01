import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from tika import parser, unpack

import re


codex_regexes = {
    re.compile(r'арбитражн[а-я]*[\s\-]+процессуальн[а-я]*\s+кодекс[а-я]*', re.IGNORECASE | re.MULTILINE): 'АПК',
    re.compile(r'гражданск[а-я]*\s+кодекс[а-я]*', re.IGNORECASE | re.MULTILINE): 'ГК',
    re.compile(r'налогов[а-я]*\s+кодекс[а-я]*', re.IGNORECASE | re.MULTILINE): 'НК',
    re.compile(r'кодекс[а-я]*\s+административного\s+судопроизводства', re.IGNORECASE | re.MULTILINE): 'КАС',
    re.compile(r'кодекс[а-я]*\s+(об\s+)?административн[а-я]*\s+правонарушени[а-я]*', re.IGNORECASE | re.MULTILINE): 'КоАП',
}

CAP_SPACES = re.compile(r'(([А-Я] ){2,}[А-Я])')


def fix_cap_spaces(text: str):
    for m in CAP_SPACES.finditer(text):
        group = m.group(1)
        text = text.replace(group, group.replace(' ', ''), 1)
    return text


def remove_newlines(text: str):
    regex = re.compile(r'([а-яА-Я,"«»()0-9])\s*\n+', re.MULTILINE)
    return regex.sub(r'\1 ', text)


def remove_numbers(text: str):
    text = re.sub(r'\d[\d ]+руб.( \d\d коп.)?', 'SUM', text)
    # text = re.sub(r'\d+', 'NUM', text)
    return text


def parse(path):
    try:
        parsed = unpack.from_file(path)  # unpack is faster in this case
    except Exception as e:
        print('Exception while reading %s: %s' % (path, e))
        return None

    for regex, repl in codex_regexes.items():
        parsed['content'] = regex.sub(repl, parsed['content'])  # TODO optimize
    parsed['content'] = fix_cap_spaces(parsed['content'])
    parsed['content'] = remove_newlines(parsed['content'])
    parsed['content'] = remove_numbers(parsed['content'])
    return parsed['content'].strip()


texts = [parse('../out/docs/' + path) for path in os.listdir('../out/docs')]

pipeline = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer())
])

