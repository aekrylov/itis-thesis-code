import re

from natasha import MoneyExtractor, OrganisationExtractor, DatesExtractor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from pymystem3 import Mystem


def make_regex(text: str):
    return re.compile(text.replace(' ', r'\s+'), re.IGNORECASE | re.MULTILINE)


m = Mystem()

codex_regexes = {
    re.compile(r'арбитражн[а-я]*[\s\-]+процессуальн[а-я]*\s+кодекс[а-я]*', re.IGNORECASE | re.MULTILINE): 'АПК',
    re.compile(r'гражданск[а-я]*\s+кодекс[а-я]*', re.IGNORECASE | re.MULTILINE): 'ГК',
    re.compile(r'налогов[а-я]*\s+кодекс[а-я]*', re.IGNORECASE | re.MULTILINE): 'НК',
    re.compile(r'кодекс[а-я]*\s+административного\s+судопроизводства', re.IGNORECASE | re.MULTILINE): 'КАС',
    re.compile(r'кодекс[а-я]*\s+(об\s+)?административн[а-я]*\s+правонарушени[а-я]*', re.IGNORECASE | re.MULTILINE): 'КоАП',
}

abbrs = [
    (make_regex(r'обществ\w+ с ограниченной ответственностью'), 'ООО'),
    (make_regex(r'открыт\w+ акционерн\w+ обществ\w+'), 'ОАО'),
    (make_regex(r'закрыт\w+ акционерн\w+ обществ\w+'), 'ЗАО'),
    (make_regex(r'публичн\w+ акционерн\w+ обществ\w+'), 'ПАО'),
    (make_regex(r'акционерн\w+ обществ\w+'), 'АО'),
    (make_regex(r'федеральн\w+ казенн\w+ учрежден\w+'), 'ФКУ'),
]

money = MoneyExtractor()
dates = DatesExtractor()
org = OrganisationExtractor()


CAP_SPACES = re.compile(r'\s((?:[А-Я]\s+){2,}[А-Я][^\w])', re.IGNORECASE)


def fix_cap_spaces(text: str):
    return CAP_SPACES.sub(lambda m: ' '+m.group(1).replace(' ', '')+' ', text)


def remove_newlines(text: str):
    regex = re.compile(r'([а-яА-Я,"«»()0-9])\s*\n+', re.MULTILINE)
    return regex.sub(r'\1 ', text)


def remove_numbers(text: str):
    # for match in reversed(money(text)):
    #     text = text[:match.span[0]] + 'SUM' + text[match.span[1]:]
    #
    # for match in reversed(dates(text)):
    #     text = text[:match.span[0]] + 'DATE' + text[match.span[1]:]
    text = re.sub(r'\d[\d\s]+([,.]\d\d\s*)?руб(\.|л[а-я]+)(\s*\d\d\s*коп(\.|[а-я]+))?', 'SUM', text)
    text = re.sub(r'\d\d?\.\d\d?\.\d{4}', 'DATE', text)
    text = re.sub(r'\d\d? [а-я]+ \d{4} г(\.|ода)', 'DATE', text)
    text = re.sub(r'[2-9]\d{3,}', 'NUM', text)  # оставляем статьи ГК, их 1551 штука
    # text = re.sub(r'\d+', 'NUM', text)
    return text


def parse_orgs_simple(text: str):
    for regex, repl in abbrs:
        text = regex.sub(repl, text)  # TODO optimize

    text = re.sub(r"[А-Я]+\s+«[^»]{3,}»", 'ORG', text)
    text = re.sub(r"[А-Я]+\s+\".*?\"(?=[^\w])", 'ORG', text)
    return text


def enum_orgs(text: str):
    names = set()
    for match in reversed(org(text)):
        name = match.fact.name
        names.add(name)
        idx = list(names).index(match.fact.name)
        text = text[:match.span[0]] + 'ORG%d' % (idx,) + text[match.span[1]:]

    return text


def cut_parts(text: str) -> str:
    text = re.sub(r'^.*установил\s*:\s*\n', '', text, 1, re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\s*решил\s*:\s*\n.*$', '', text, 1, re.IGNORECASE | re.DOTALL)
    return text


def preprocess(text: str) -> str:
    text = text.strip()
    text = fix_cap_spaces(text)
    text = cut_parts(text)

    for regex, repl in codex_regexes.items():
        text = regex.sub(repl, text)  # TODO optimize

    text = remove_newlines(text)
    text = remove_numbers(text)
    text = parse_orgs_simple(text)
    # text = enum_orgs(text)
    return text


pipeline = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer())
])

