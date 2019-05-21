import json
import sys
from collections import defaultdict

import numpy as np
from gensim.models import TfidfModel

from recommenders.db import Rating
from recommenders.util import load_uci


class Evaluator:

    top_n = 20

    def __init__(self, cut_off=20):
        ratings = Rating.query.all()
        self.test_data = defaultdict(lambda: defaultdict(int))
        for r in ratings:
            self.test_data[r.doc_id][r.recommendation_id] = r.value
            #  relevance is usually reciprocal
            self.test_data[r.recommendation_id].setdefault(r.doc_id, r.value)

        # Remove test entries with too little recommendation data
        for k in list(self.test_data.keys()):
            if len(self.test_data[k]) < cut_off:
                del self.test_data[k]

    def evaluate(self, model, corpus):
        return {
            'map': self.map_score(model, corpus),
            'map_known': self.map_score_known(model, corpus)
        }

    @staticmethod
    def precision(recs: [int], test_scores):
        return np.average([test_scores[rec] > 0 for rec in recs])

    @staticmethod
    def average_precision_score(recs: [int], test_scores: dict, remove_unknown=False):
        k_values = range(1, len(recs))
        if remove_unknown:
            # Remove values for which rel(k) is unknown
            k_values = [k for k in k_values if test_scores[recs[k-1]] != 0]

        precisions = [Evaluator.precision(recs[:k], test_scores) for k in k_values]
        rel = [test_scores[recs[k-1]] > 0 for k in k_values]
        return np.multiply(precisions, rel).mean()

    def map_score(self, model, corpus):
        """Mean Average Precision score"""
        return np.average([
            self.average_precision_score(model.get_similar(corpus[k], self.top_n), v) for k, v in self.test_data.items()
        ])

    def map_score_known(self, model, corpus):
        """mAP score only for known items"""
        return np.average([
            self.average_precision_score(model.get_similar(corpus[k], self.top_n), v, True) for k, v in self.test_data.items()
        ])


if __name__ == '__main__':
    from recommenders.models import LsiModel, LdaModel, Doc2vecModel, BigArtmModel
    import recommenders.webapp_config as conf

    lsi_on = '--lsi' in sys.argv
    lda_on = '--lda' in sys.argv
    d2v_on = '--d2v' in sys.argv
    artm_on = '--artm' in sys.argv
    trained = '--trained' in sys.argv

    evaluator = Evaluator()
    scores = defaultdict(dict)

    corpus, data_samples, dictionary, metadata = load_uci(conf.DOCS_LOCATION)
    tfidf = TfidfModel(dictionary=dictionary, smartirs='ntc')
    corpus_raw = corpus
    corpus = [tfidf[doc] for doc in corpus]

    if trained:
        import recommenders.webapp as webapp

        for k in ['lsi', 'lda', 'artm', 'artm2']:
            scores['t'+k] = evaluator.evaluate(getattr(webapp, k), corpus)
        scores['t_d2v'] = evaluator.evaluate(webapp.d2v, data_samples)
        scores['t_artm_raw'] = evaluator.evaluate(webapp.artm, corpus_raw)
        scores['t_artm2_raw'] = evaluator.evaluate(webapp.artm2, corpus_raw)

        with open('./scores_trained.json', 'w') as f:
            json.dump(scores, f)

    for t in range(50, 800, 50):
        if lsi_on:
            scores['lsi'][t] = evaluator.evaluate(LsiModel(corpus, dictionary, t), corpus)
        if lda_on:
            scores['lda'][t] = evaluator.evaluate(LdaModel(corpus, dictionary, t), corpus)
        if d2v_on:
            scores['d2v'][t] = evaluator.evaluate(Doc2vecModel(data_samples, t), corpus)
        if artm_on:
            scores['artm'][t] = evaluator.evaluate(BigArtmModel(conf.UCI_FOLDER, dictionary, t), corpus)

    print(scores)
    with open('./scores.json', 'w') as f:
        json.dump(scores, f)
