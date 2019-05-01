import itertools
import pickle
import sys
from time import time

from gensim.corpora import UciCorpus, Dictionary

from recommenders.models import Tokenizer
from text_processing.simple import parse_all, cache_path


def load_corpus(location, n_samples=None):
    print("Loading the corpus...")
    t0 = time()
    data_samples = list(itertools.islice(parse_all(location, from_cache=True, with_paths=True), n_samples))
    paths, data_samples = zip(*data_samples)
    paths = [cache_path(p) for p in paths]
    print("loaded %d samples in %0.3fs." % (len(data_samples), time() - t0))
    return paths, data_samples


def vectorize(corpus):
    tokenized = [Tokenizer.tokenize(doc) for doc in corpus]

    dictionary = Dictionary(tokenized)
    dictionary.filter_extremes(no_below=10, no_above=0.66)
    bows = [dictionary.doc2bow(doc) for doc in tokenized]
    return dictionary, bows


def save_uci(paths, corpus, dictionary, location):
    UciCorpus.serialize(location, corpus, id2word=dictionary)
    with open(location + '.docs.pickle', 'wb') as f:
        pickle.dump(paths, f)


if __name__ == '__main__':
    corpus_location = sys.argv[-2] if len(sys.argv) > 2 else '../out/docs_simple2'
    save_location = sys.argv[-1] if len(sys.argv) > 1 else '../out/corpus.uci'

    paths, data_samples = load_corpus(corpus_location)
    dictionary, docs = vectorize(data_samples)

    save_uci(paths, docs, dictionary, save_location)

