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


class ModelBase:
    STOP_WORDS = {'от', 'на', 'не', 'рф', 'ст'}
    stemmer = SnowballStemmer("russian")
    CACHE = key_dependent_dict(lambda w: LsiModel.stemmer.stem(w))
    analyzer = TfidfVectorizer().build_analyzer()

    @staticmethod
    def tokenize(doc: str):
        return [LsiModel.CACHE[w] for w in LsiModel.analyzer(doc) if w not in LsiModel.STOP_WORDS]

    @staticmethod
    def vectorize(corpus):
        tokenized = [LsiModel.tokenize(doc) for doc in corpus]

        dictionary = corpora.Dictionary(tokenized)
        dictionary.filter_extremes(no_below=10, no_above=0.66)
        bows = [dictionary.doc2bow(doc) for doc in tokenized]
        return dictionary, bows


class LsiModel(ModelBase):

    def __init__(self, corpus, dictionary, n_topics):
        self.dictionary = dictionary

        print('Building the index')
        t0 = time()
        self.tfidf = models.TfidfModel(corpus, smartirs='ntc')
        corpus_tfidf = self.tfidf[corpus]

        self.lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)
        self.corpus_lsi = self.lsi[corpus_tfidf]
        self.index = similarities.MatrixSimilarity(self.corpus_lsi, num_features=n_topics)
        print("TF-DF + LSI built in %.3fs" % (time() - t0))

    def get_similar_ids(self, idx):
        vec_lsi = self.corpus_lsi[idx]
        sims = self.index[vec_lsi]
        return np.argsort(-sims)[1:]

    def get_similar_for_text(self, text):
        doc = self.dictionary.doc2bow(self.tokenize(text))
        vec_lsi = self.lsi[self.tfidf[doc]]
        sims = self.index[vec_lsi]
        return np.argsort(-sims)[1:]


class LdaModel(ModelBase):

    def __init__(self, corpus, dictionary, n_topics):
        self.dictionary = dictionary

        print('Building the index')
        t0 = time()
        self.tfidf = models.TfidfModel(corpus, smartirs='ntc')
        corpus_tfidf = self.tfidf[corpus]

        self.lda = models.LdaMulticore(corpus_tfidf, id2word=dictionary, num_topics=n_topics, workers=1)
        self.corpus_lda = self.lda[corpus_tfidf]
        self.index = similarities.MatrixSimilarity(self.corpus_lda, num_features=n_topics)
        print("TF-DF + LDA built in %.3fs" % (time() - t0))

    def get_similar_ids(self, idx):
        vec_lsi = self.corpus_lda[idx]
        sims = self.index[vec_lsi]
        return np.argsort(-sims)[1:]

    def get_similar_for_text(self, text):
        doc = self.dictionary.doc2bow(self.tokenize(text))
        vec_lsi = self.lda[self.tfidf[doc]]
        sims = self.index[vec_lsi]
        return np.argsort(-sims)[1:]