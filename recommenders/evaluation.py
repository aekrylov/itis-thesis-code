import os
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

        # Remove test entries with too little recommendation data
        for k in list(self.test_data.keys()):
            if len(self.test_data[k]) < cut_off:
                del self.test_data[k]

    def evaluate(self, model, corpus):
        return {
            'map': self.map_score(model, corpus)
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
    from recommenders.models import LsiModel

    corpus, data_samples, dictionary, metadata = load_uci(os.environ['DOCS_LOCATION'])
    tfidf = TfidfModel(dictionary=dictionary, smartirs='ntc')
    corpus = [tfidf[doc] for doc in corpus]

    evaluator = Evaluator()

    for t in range(50, 500, 50):
        lsi = LsiModel(corpus, dictionary, t)
        print('mAP for k=%d: %.4f' % (t, evaluator.map_score(lsi, corpus)))
        print('mAP-known for k=%d: %.4f' % (t, evaluator.map_score_known(lsi, corpus)))
