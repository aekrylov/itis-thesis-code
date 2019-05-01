import logging
from collections import defaultdict
from time import time

import numpy as np
from gensim import models, similarities
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


class SimilarityIndex(ModelBase):

    def __init__(self, corpus, model, n_topics):
        self.model = model

        print('Building the index')
        t0 = time()
        self.index = similarities.MatrixSimilarity(model[corpus], num_features=n_topics, num_best=self.N_BEST)
        print("Index built in %.3fs" % (time() - t0))

    def get_similar(self, doc, topn=10):
        sims = self.index[self.model[doc]]
        return np.argsort(-sims)[:topn]


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
        return np.argsort(-sims)[:topn]


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
        return np.argsort(-sims)[:topn]


class BigArtmModel(ModelBase):

    def __init__(self, uci_dir, dictionary, n_topics):
        bv = artm.BatchVectorizer(data_format='bow_uci', data_path=uci_dir, collection_name='corpus',
                                  target_folder=uci_dir + '/artm_batches')
        bv_dict = bv.dictionary
        topic_names = [str(i) for i in range(n_topics)]

        logging.info("Fitting the ARTM model")
        model = artm.ARTM(topic_names=topic_names, dictionary=bv_dict, cache_theta=True,
                          scores=[
                                   artm.PerplexityScore(name='PerplexityScore', dictionary=bv_dict),
                                   artm.SparsityPhiScore(name='SparsityPhiScore'),
                                   artm.SparsityThetaScore(name='SparsityThetaScore'),
                                   artm.TopicKernelScore(name='TopicKernelScore', probability_mass_threshold=0.3)
                          ],
                          regularizers=[
                                   artm.SmoothSparseThetaRegularizer(name='SparseTheta', tau=-0.15),
                                   artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=-0.1),
                                   artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=1.5e+5)
                          ])

        model.fit_offline(batch_vectorizer=bv, num_collection_passes=10)

        logging.info("Processing word-topic matrices")

        # Create a new word-topic matrix according to dictionary indices
        self.phi = np.zeros((len(dictionary), n_topics), dtype=np.float64)
        for word, vec in model.phi_.iterrows():
            idx = dictionary.token2id[word[1]]
            self.phi[idx, :] = vec

        logging.info("Building the index for ARTM")
        corpus = model.get_theta().T.sort_index()
        self.index = similarities.MatrixSimilarity(corpus, num_features=n_topics, num_best=self.N_BEST)

    def get_similar(self, doc, topn=10):
        vec = np.matmul([doc], self.phi)
        sims = self.index[vec]
        return np.argsort(-sims)[:topn]


class Doc2vecModel(ModelBase):
    """
    NB: works with the raw corpus
    """
    def __init__(self, data_samples, n_topics, window=10):
        doc2vec_corpus = [TaggedDocument(Tokenizer.tokenize(sample), [i]) for i, sample in enumerate(data_samples)]
        self.model = Doc2Vec(doc2vec_corpus, vector_size=n_topics, window=window, min_count=10, workers=4)
        self.model.delete_temporary_training_data()

    def get_similar(self, doc, topn=10):
        if isinstance(doc, str):
            doc = Tokenizer.tokenize(doc)

        return [t[0] for t in self.model.docvecs.most_similar(positive=[self.model.infer_vector(doc)], topn=topn)]
