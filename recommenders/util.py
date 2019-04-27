import itertools
import os
import pickle
from time import time

from gensim.corpora import TextDirectoryCorpus, UciCorpus
from gensim.models import TfidfModel
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from recommenders.models import LsiModel, key_dependent_dict
from text_processing.simple import parse_all, cache_path


class TextDirectoryLimitInMemoryCorpus(TextDirectoryCorpus):
    STOP_WORDS = {'от', 'на', 'не', 'рф', 'ст'}
    stemmer = SnowballStemmer("russian")
    CACHE = key_dependent_dict(lambda w: LsiModel.stemmer.stem(w))
    analyzer = TfidfVectorizer().build_analyzer()

    @staticmethod
    def tokenize(doc: str):
        return [__class__.CACHE[w] for w in __class__.analyzer(doc) if w not in __class__.STOP_WORDS]

    def __init__(self, input, limit=None, dictionary=None, metadata=False, min_depth=0, max_depth=None, pattern=None,
                 exclude_pattern=None, lines_are_documents=False, **kwargs):
        self.limit = limit
        self._cache = None
        super().__init__(input, dictionary, metadata, min_depth, max_depth, pattern, exclude_pattern,
                         lines_are_documents, token_filters=[], character_filters=[], **kwargs)

    def getstream(self):
        if self._cache is None:
            stream = itertools.islice(super().getstream(), self.limit)
            self._cache = list(stream)
            if not self.length:
                self.length = len(self._cache)
        return self._cache

    def getlist(self):
        return self.getstream()

    def preprocess_text(self, text):
        return self.tokenize(text)


def load_corpus(location, n_samples=None):
    print("Loading the corpus...")
    t0 = time()
    data_samples = list(itertools.islice(parse_all(location, from_cache=True), n_samples))
    print("loaded %d samples in %0.3fs." % (len(data_samples), time() - t0))
    return data_samples


def load_corpus2(location, n_samples=None):
    print("Loading the corpus...")
    t0 = time()
    corpus = TextDirectoryLimitInMemoryCorpus(cache_path(location), limit=n_samples)
    print("loaded %d samples in %0.3fs." % (len(corpus), time() - t0))
    return corpus


def load_uci(location):
    print("Loading the corpus...")
    t0 = time()
    paths = pickle.load(open(location + '.docs.pickle', 'rb'))
    data_samples = [''.join(open(p, 'r')) for p in paths]
    corpus = UciCorpus(location)
    print("loaded %d samples in %0.3fs." % (len(corpus), time() - t0))
    return corpus, data_samples


def process_tfidf(corpus, dictionary):
    model = TfidfModel(dictionary=dictionary, smartirs='ntc')
    return [model[doc] for doc in corpus]


def load_model(constructor, pickle_path=None):
    if pickle_path and os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    else:
        model = constructor()
        if pickle_path:
            with open(pickle_path, 'wb') as f:
                pickle.dump(model, f)
        return model
