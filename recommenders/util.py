import json
import os
import pickle
import re
from time import time

from gensim.corpora import UciCorpus

from recommenders.models import Tokenizer


class RandomAccessCorpus:
    """
    Enables random access for a filesystem-based corpus
    """
    def __init__(self, paths):
        self.paths = paths
        self.len = len(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        with open(self.paths[item], 'r') as f:
            return ''.join(f)

    def __iter__(self):
        for i in range(self.len):
            yield self[i]


def load_uci(location):
    print("Loading the corpus...")
    t0 = time()

    paths = load(location + '.docs.pickle')

    dictionary = load(location + '.dict.pickle')

    with open(location + '.meta.json', 'r') as f_meta:
        meta_map = {}
        for line in f_meta:
            doc_meta = json.loads(line)
            meta_map[doc_meta['case_id']] = doc_meta
        metadata = [meta_map[re.search(r'([a-z0-9-]+)\.txt', p).group(1)] for p in paths]

    data_samples = RandomAccessCorpus(paths)
    corpus = UciCorpus(location)
    print("loaded %d samples in %0.3fs." % (len(corpus), time() - t0))

    return corpus, data_samples, dictionary, metadata


def tokenize(text, dictionary):
    return dictionary.doc2bow(Tokenizer.tokenize(text))


def load(pickle_path):
    if pickle_path is None or not os.path.exists(pickle_path):
        raise FileNotFoundError()
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def kad_pdf_path(meta):
    return 'http://kad.arbitr.ru/PdfDocument/%s/%s/%s' % (meta['case_id'], meta['doc_id'], meta['doc_name'])
