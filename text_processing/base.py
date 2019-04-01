import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

codex_regexes = {
    re.compile(r'арбитражн[а-я]*[\s\-]+процессуальн[а-я]*\s+кодекс[а-я]*', re.IGNORECASE | re.MULTILINE): 'АПК',
    re.compile(r'гражданск[а-я]*\s+кодекс[а-я]*', re.IGNORECASE | re.MULTILINE): 'ГК',
    re.compile(r'налогов[а-я]*\s+кодекс[а-я]*', re.IGNORECASE | re.MULTILINE): 'НК',
    re.compile(r'кодекс[а-я]*\s+административного\s+судопроизводства', re.IGNORECASE | re.MULTILINE): 'КАС',
    re.compile(r'кодекс[а-я]*\s+(об\s+)?административн[а-я]*\s+правонарушени[а-я]*', re.IGNORECASE | re.MULTILINE): 'КоАП',
}

CAP_SPACES = re.compile(r'(([А-Я] +){2,}[А-Я])')


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


def cut_parts(text: str) -> str:
    no_head = re.sub(r'^.*\n\s*установил:\s*\n', '', text, 1, re.MULTILINE | re.IGNORECASE | re.DOTALL)
    no_resolution = re.sub(r'\n\s*решил:\s*\n.*$', '', no_head, 1, re.MULTILINE | re.IGNORECASE | re.DOTALL)
    return no_resolution


def preprocess(text: str) -> str:
    text = text.strip()
    text = fix_cap_spaces(text)
    text = cut_parts(text)

    for regex, repl in codex_regexes.items():
        text = regex.sub(repl, text)  # TODO optimize

    text = remove_newlines(text)
    text = remove_numbers(text)
    return text


pipeline = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer())
])

