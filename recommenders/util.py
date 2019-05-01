import os
import pickle
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
    paths = pickle.load(open(location + '.docs.pickle', 'rb'))
    data_samples = RandomAccessCorpus(paths)
    corpus = UciCorpus(location)
    print("loaded %d samples in %0.3fs." % (len(corpus), time() - t0))
    return corpus, data_samples


def tokenize(text, dictionary):
    return dictionary.doc2bow(Tokenizer.tokenize(text))


def load(pickle_path):
    if pickle_path is None or not os.path.exists(pickle_path):
        raise FileNotFoundError()
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)
