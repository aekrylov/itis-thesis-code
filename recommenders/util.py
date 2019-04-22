import itertools
import os
import pickle
from time import time

from recommenders.lsi import LsiModel
from text_processing.simple import parse_all


def load_corpus(location, n_samples):
    print("Loading the corpus...")
    t0 = time()
    data_samples = list(itertools.islice(parse_all(location, from_cache=True), n_samples))
    print("loaded %d samples in %0.3fs." % (len(data_samples), time() - t0))
    return data_samples


def load_lsi(data_samples, n_topics, pickle_path=None):
    if pickle_path and os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    else:
        model = LsiModel(data_samples, n_topics)
        if pickle_path:
            with open(pickle_path, 'wb') as f:
                pickle.dump(model, f)
        return model