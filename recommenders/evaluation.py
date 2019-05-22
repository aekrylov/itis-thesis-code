import json
import sys
from collections import defaultdict

import math
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
            'map_known': self.map_score_known(model, corpus),
            'mean_p_at_k': self.mean_precision_at_k(model, corpus),
            'mean_p_at_k_known': self.mean_precision_at_k(model, corpus, True),
            'mean_dcg': self.mean_dcg(model, corpus),
            'mean_dcg_known': self.mean_dcg(model, corpus, True),
        }

    @staticmethod
    def precision(recs: [int], test_scores, remove_unknown=False):
        if remove_unknown:
            recs = [rec for rec in recs if test_scores[rec] != 0]
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

    @staticmethod
    def dcg(recs: [int], test_scores, remove_unknown=False):
        if remove_unknown:
            recs = [rec for rec in recs if test_scores[rec] != 0]
        return sum([(test_scores[rec] > 0) / math.log2(i+2) for i,rec in enumerate(recs)])

    def mean_precision_at_k(self, model, corpus, remove_unknown=False):
        a = np.asarray([
            self.precision(model.get_similar(corpus[k], self.top_n), v, remove_unknown)
            for k, v in self.test_data.items()
        ])
        return a[a == a].mean()  # filter out nan values

    def map_score(self, model, corpus):
        """Mean Average Precision score"""
        return np.average([
            self.average_precision_score(model.get_similar(corpus[k], self.top_n), v) for k, v in self.test_data.items()
        ])

    def map_score_known(self, model, corpus):
        """mAP score only for known items"""
        ap_scores = np.asarray([
            self.average_precision_score(model.get_similar(corpus[k], self.top_n), v, True)
            for k, v in self.test_data.items()
        ])
        return ap_scores[ap_scores == ap_scores].mean()  # filter out nan values

    def mean_dcg(self, model, corpus, remove_unknown=False):
        a = np.asarray([
            self.dcg(model.get_similar(corpus[k], self.top_n), v, remove_unknown)
            for k, v in self.test_data.items()
        ])
        return a[a == a].mean()  # filter out nan values


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

        for k in ['lsi', 'lda', 'artm']:
            scores['t'+k] = evaluator.evaluate(getattr(webapp, k), corpus)
        scores['t_d2v'] = evaluator.evaluate(webapp.d2v, data_samples)
        scores['t_artm_raw'] = evaluator.evaluate(webapp.artm, corpus_raw)

        with open('./scores_trained.json', 'w') as f:
            json.dump(scores, f)

    t_values = [2**i for i in range(2, 8)] + list(range(100, 850, 50))
    for t in t_values:
        if lsi_on:
            scores['lsi'][t] = evaluator.evaluate(LsiModel(corpus, dictionary, t), corpus)
        if lda_on:
            scores['lda'][t] = evaluator.evaluate(LdaModel(corpus, dictionary, t), corpus)
        if d2v_on:
            scores['d2v'][t] = evaluator.evaluate(Doc2vecModel(data_samples, t), data_samples)
        if artm_on:
            scores['artm'][t] = evaluator.evaluate(BigArtmModel(conf.UCI_FOLDER, dictionary, t), corpus)

    print(scores)
    with open('./scores.json', 'w') as f:
        json.dump(scores, f)
