import itertools
from time import time

from gensim.corpora import UciCorpus, Dictionary
from gensim.models import TfidfModel

from recommenders.models import ModelBase
from text_processing.simple import parse_all


def load_corpus(location, n_samples=None):
    print("Loading the corpus...")
    t0 = time()
    data_samples = list(itertools.islice(parse_all(location, from_cache=True), n_samples))
    print("loaded %d samples in %0.3fs." % (len(data_samples), time() - t0))
    return data_samples


def vectorize(corpus):
    tokenized = [ModelBase.tokenize(doc) for doc in corpus]

    dictionary = Dictionary(tokenized)
    dictionary.filter_extremes(no_below=10, no_above=0.66)
    bows = [dictionary.doc2bow(doc) for doc in tokenized]
    return dictionary, bows


def process_tfidf(corpus, dictionary):
    model = TfidfModel(dictionary=dictionary, smartirs='ntc')
    return [model[doc] for doc in corpus]


def save_uci(corpus, dictionary, location):
    UciCorpus.serialize(location, corpus, id2word=dictionary)


corpus_location = '../out/docs_simple2'
save_location = '../out/corpus.uci'

data_samples = load_corpus(corpus_location)
dictionary, docs = vectorize(data_samples)

save_uci(docs, dictionary, save_location)
