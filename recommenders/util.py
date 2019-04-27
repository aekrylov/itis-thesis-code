import os
import pickle
from time import time

from gensim.corpora import UciCorpus
from gensim.models import TfidfModel


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
