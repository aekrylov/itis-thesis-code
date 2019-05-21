import logging
import pickle
import sys

from gensim.models import TfidfModel

import recommenders.webapp_config as conf
from recommenders.models import LsiModel, LdaModel, Doc2vecModel, BigArtmModel
from recommenders.util import load_uci


def save(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    lsi_on = '--lsi' in sys.argv
    lda_on = '--lda' in sys.argv
    d2v_on = '--d2v' in sys.argv
    artm_on = '--artm' in sys.argv

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    corpus, data_samples, dictionary, _ = load_uci(conf.DOCS_LOCATION)

    if lsi_on or lda_on:
        tfidf = TfidfModel(dictionary=dictionary, smartirs='ntc')
        corpus = [tfidf[doc] for doc in corpus]

    if lsi_on:
        LsiModel(corpus, dictionary, conf.N_TOPICS).save(conf.LSI_PICKLE)

    if lda_on:
        LdaModel(corpus, dictionary, conf.N_TOPICS).save(conf.LDA_PICKLE)

    if d2v_on:
        Doc2vecModel(data_samples, conf.N_TOPICS).save(conf.D2V_PICKLE)

    if artm_on:
        BigArtmModel(conf.UCI_FOLDER, dictionary, conf.N_TOPICS).save(conf.ARTM_PICKLE)
