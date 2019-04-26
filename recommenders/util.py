import itertools
import os
import pickle
from time import time

from gensim.corpora import TextDirectoryCorpus

from recommenders.models import LsiModel, LdaModel
from text_processing.simple import parse_all, cache_path


class TextDirectoryLimitInMemoryCorpus(TextDirectoryCorpus):

    def __init__(self, input, limit=None, dictionary=None, metadata=False, min_depth=0, max_depth=None, pattern=None,
                 exclude_pattern=None, lines_are_documents=False, **kwargs):
        self.limit = limit
        self._cache = None
        self._text_cache = None
        super().__init__(input, dictionary, metadata, min_depth, max_depth, pattern, exclude_pattern,
                         lines_are_documents, **kwargs)

    def getstream(self):
        if not self._cache:
            stream = itertools.islice(super().getstream(), self.limit)
            self._cache = list(stream)
            if not self.length:
                self.length = self.limit
        return self._cache

    def get_texts(self):
        if not self._text_cache:
            self._text_cache = list(super().get_texts())
        return self._text_cache

    def __getitem__(self, item):
        return self.get_texts()[item]


def load_corpus2(location, n_samples):
    print("Loading the corpus...")
    t0 = time()
    corpus = TextDirectoryLimitInMemoryCorpus(cache_path(location), limit=n_samples, tokenizer=LsiModel.tokenize)
    print("loaded %d samples in %0.3fs." % (len(corpus), time() - t0))
    return corpus


def load_lsi(corpus, n_topics, pickle_path=None):
    if pickle_path and os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    else:
        model = LsiModel(corpus, n_topics)
        if pickle_path:
            with open(pickle_path, 'wb') as f:
                pickle.dump(model, f)
        return model


def load_lda(corpus, n_topics, pickle_path=None):
    if pickle_path and os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    else:
        model = LdaModel(corpus, n_topics)
        if pickle_path:
            with open(pickle_path, 'wb') as f:
                pickle.dump(model, f)
        return model
