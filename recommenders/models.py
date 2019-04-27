from collections import defaultdict
from time import time

import numpy as np
from gensim import models, similarities, corpora
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


class key_dependent_dict(defaultdict):
    def __init__(self, f_of_x):
        super().__init__(None)  # base class doesn't get a factory
        self.f_of_x = f_of_x  # save f(x)

    def __missing__(self, key):  # called when a default needed
        ret = self.f_of_x(key)  # calculate default value
        self[key] = ret  # and install it in the dict
        return ret


class Tokenizer:
    STOP_WORDS = {'от', 'на', 'не', 'рф', 'ст'}
    stemmer = SnowballStemmer("russian")
    CACHE = key_dependent_dict(lambda w: Tokenizer.stemmer.stem(w))
    analyzer = TfidfVectorizer().build_analyzer()

    @staticmethod
    def tokenize(doc: str):
        return [Tokenizer.CACHE[w] for w in Tokenizer.analyzer(doc) if w not in Tokenizer.STOP_WORDS]


class ModelBase:

    def get_similar(self, doc):
        raise NotImplementedError()


class SimilarityIndex(ModelBase):

    def __init__(self, corpus, model, n_topics):
        self.model = model

        print('Building the index')
        t0 = time()
        self.index = similarities.MatrixSimilarity(model[corpus], num_features=n_topics)
        print("Index built in %.3fs" % (time() - t0))

    def get_similar(self, doc):
        sims = self.index[self.model[doc]]
        return np.argsort(-sims)[1:]


class LsiModel(ModelBase):

    def __init__(self, corpus, dictionary, n_topics):
        print('Building the index')
        t0 = time()
        self.lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=n_topics, chunksize=40000)
        print("LSI built in %.3fs" % (time() - t0))

        self.index = similarities.MatrixSimilarity(self.lsi[corpus], num_features=n_topics)
        print("LSI + index built in %.3fs" % (time() - t0))

    def get_similar(self, doc):
        sims = self.index[self.lsi[doc]]
        return np.argsort(-sims)[1:]


class LdaModel(ModelBase):

    def __init__(self, corpus, dictionary, n_topics):
        print('Building the index')
        t0 = time()
        self.lda = models.LdaMulticore(corpus, id2word=dictionary, num_topics=n_topics, workers=1,
                                       chunksize=4000)
        print("LDA built in %.3fs" % (time() - t0))

        self.index = similarities.MatrixSimilarity(self.lda[corpus], num_features=n_topics)
        print("LDA + index built in %.3fs" % (time() - t0))

    def get_similar(self, doc):
        sims = self.index[self.lda[doc]]
        return np.argsort(-sims)[1:]