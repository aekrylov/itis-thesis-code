import logging
import os
import pickle
from collections import defaultdict
from time import time

import numpy as np
from gensim import models, similarities, matutils
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import artm
except ImportError:
    logging.warn('ARTM module not available')


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

    N_BEST = 100

    def get_similar(self, doc, topn=10):
        raise NotImplementedError()

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path, **kwargs):
        with open(path, 'rb') as f:
            return pickle.load(f)


class SimilarityIndex(ModelBase):

    def __init__(self, corpus, model, n_topics):
        self.model = model

        print('Building the index')
        t0 = time()
        self.index = similarities.MatrixSimilarity(model[corpus], num_features=n_topics, num_best=self.N_BEST)
        print("Index built in %.3fs" % (time() - t0))

    def get_similar(self, doc, topn=10):
        sims = self.index[self.model[doc]]
        return [t[0] for t in sims[:topn]]


class LsiModel(ModelBase):

    def __init__(self, corpus, dictionary, n_topics):
        print('Building the index')
        t0 = time()
        self.lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=n_topics, chunksize=40000)
        print("LSI built in %.3fs" % (time() - t0))

        self.index = similarities.MatrixSimilarity(self.lsi[corpus], num_features=n_topics, num_best=self.N_BEST)
        print("LSI + index built in %.3fs" % (time() - t0))

    def get_similar(self, doc, topn=10):
        sims = self.index[self.lsi[doc]]
        return [t[0] for t in sims[:topn]]


class LdaModel(ModelBase):

    def __init__(self, corpus, dictionary, n_topics):
        print('Building the index')
        t0 = time()
        self.lda = models.LdaModel(corpus, id2word=dictionary, num_topics=n_topics, chunksize=4000)
        print("LDA built in %.3fs" % (time() - t0))

        self.index = similarities.MatrixSimilarity(self.lda[corpus], num_features=n_topics, num_best=self.N_BEST)
        print("LDA + index built in %.3fs" % (time() - t0))

    def get_similar(self, doc, topn=10):
        sims = self.index[self.lda[doc]]
        return [t[0] for t in sims[:topn]]


class BigArtmModel(ModelBase):

    def __init__(self, uci_dir, dictionary, n_topics):
        bv = artm.BatchVectorizer(data_format='bow_uci', data_path=uci_dir, collection_name='corpus',
                                  target_folder=uci_dir + '/artm_batches')
        bv_dict = bv.dictionary

        logging.info("Fitting the ARTM model")
        model = artm.ARTM(dictionary=bv_dict, num_topics=n_topics)

        model.fit_offline(batch_vectorizer=bv, num_collection_passes=10)

        logging.info("Processing word-topic matrices")

        # Create a new word-topic matrix according to dictionary indices
        self.phi = np.zeros(model.phi_.shape, dtype=np.float64)
        for word, vec in model.phi_.iterrows():
            idx = dictionary.token2id[word[1]]
            self.phi[idx, :] = vec

        logging.info("Building the index for ARTM")
        corpus = model.transform(bv).T.sort_index()
        corpus = [matutils.full2sparse(row) for index, row in corpus.iterrows()]
        self.index = similarities.MatrixSimilarity(corpus, num_features=n_topics, num_best=self.N_BEST)

        self.model = model
        self.dictionary = dictionary

    def save(self, path):
        super().save(path)
        self.model.save(path+'.artm')

    @staticmethod
    def load(path, **kwargs):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            obj.model = artm.ARTM(num_topics=10)
            obj.model.load(path+'.artm')
            return obj

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['model']
        return d

    def get_similar(self, doc, topn=10):
        m = np.asarray([matutils.sparse2full(doc, len(self.dictionary))])
        bv = artm.BatchVectorizer(data_format='bow_n_wd', n_wd=m.T, vocabulary=self.dictionary)
        sims = self.index[matutils.full2sparse(self.model.transform(bv))]
        return [t[0] for t in sims[:topn]]


class Doc2vecModel(ModelBase):
    """
    NB: works with the raw corpus
    """
    def __init__(self, data_samples, n_topics, window=10):
        docs = [TaggedDocument(Tokenizer.tokenize(sample), [i]) for i, sample in enumerate(data_samples)]
        self.model = Doc2Vec(docs, vector_size=n_topics, window=window, min_count=10, workers=os.cpu_count())
        self.model.delete_temporary_training_data()

    def get_similar(self, doc, topn=10):
        if isinstance(doc, str):
            doc = Tokenizer.tokenize(doc)

        return [t[0] for t in self.model.docvecs.most_similar(positive=[self.model.infer_vector(doc)], topn=topn)]
