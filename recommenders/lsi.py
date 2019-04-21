import itertools
from collections import defaultdict
from time import time

import numpy as np
from gensim import corpora, models, similarities
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from text_processing.simple import parse_all


class key_dependent_dict(defaultdict):
    def __init__(self, f_of_x):
        super().__init__(None)  # base class doesn't get a factory
        self.f_of_x = f_of_x  # save f(x)

    def __missing__(self, key):  # called when a default needed
        ret = self.f_of_x(key)  # calculate default value
        self[key] = ret  # and install it in the dict
        return ret


class LsiModel:

    STOP_WORDS = {'от', 'на', 'не', 'рф', 'ст'}
    stemmer = SnowballStemmer("russian")
    CACHE = key_dependent_dict(lambda w: LsiModel.stemmer.stem(w))
    analyzer = TfidfVectorizer().build_analyzer()

    def __init__(self, data_samples, n_topics):
        print("Vectorizing the corpus")
        t0 = time()
        dictionary, corpus = self.vectorize(data_samples)
        print('Done in %.3fs' % (time() - t0))

        print('Building the index')
        t0 = time()
        tfidf = models.TfidfModel(corpus, smartirs='nnc')
        corpus_tfidf = tfidf[corpus]

        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)
        corpus_lsi = lsi[corpus_tfidf]
        index = similarities.MatrixSimilarity(corpus_lsi)
        print("TF-DF + LSI built in %.3fs" % (time() - t0))

        self.dictionary = dictionary
        self.corpus_lsi = corpus_lsi
        self.index = index

    def get_similar_ids(self, idx):
        vec_lsi = self.corpus_lsi[idx]
        sims = self.index[vec_lsi]  # perform a similarity vector against the corpus
        return np.argsort(-sims)[1:]

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


n_samples = 20000

print("Loading the corpus...")
t0 = time()
data_samples = list(itertools.islice(parse_all("../out/docs_simple2", from_cache=True), n_samples))
print("loaded %d samples in %0.3fs." % (len(data_samples), time() - t0))

model = LsiModel(data_samples, 800)

idx = 215
print(data_samples[idx])
most_similar = model.get_similar_ids(idx)
print()
print(most_similar[0])
print()
print(data_samples[most_similar[0]])


