import logging
import random
import tempfile

from flask import Flask, render_template, request
from flask_restful import Api, Resource
from flask_sqlalchemy import SQLAlchemy
from gensim.models import TfidfModel
from tika import unpack

from recommenders.util import load_uci, tokenize, load, kad_pdf_path
from text_processing.base import preprocess


def doc_for_api(doc_id):
    return {
        'id': int(doc_id),
        'url': api.url_for(DocResource, doc_id=doc_id, _external=True),
        'case_number': metadata[doc_id]['case_num'],
        'pdf_path': kad_pdf_path(metadata[doc_id]),
    }


def get_similar(data_sample, idx_to_doc=lambda x: x, cut_first=False):
    bow = tokenize(data_sample, dictionary)
    vec_tfidf = tfidf[bow]

    start_idx = 1 if cut_first else 0
    return {
        'lsi': [idx_to_doc(sim) for sim in lsi.get_similar(vec_tfidf)[start_idx:]],
        'lda': [idx_to_doc(sim) for sim in lda.get_similar(vec_tfidf)[start_idx:]],
        'd2v': [idx_to_doc(sim) for sim in d2v.get_similar(data_sample)[start_idx:]],
        'artm': [idx_to_doc(sim) for sim in artm.get_similar(bow)[start_idx:]],
        'artm_tfidf': [idx_to_doc(sim) for sim in artm.get_similar(vec_tfidf)[start_idx:]],
        'artm2': [idx_to_doc(sim) for sim in artm2.get_similar(bow)[start_idx:]],
        'artm2_tfidf': [idx_to_doc(sim) for sim in artm2.get_similar(vec_tfidf)[start_idx:]],
    }


class UploadResource(Resource):
    def post(self):
        file = request.files['file']

        tmp_file = tempfile.NamedTemporaryFile()
        file.save(tmp_file)
        text = preprocess(unpack.from_file(tmp_file.name)['content'])
        tmp_file.close()

        return {
            'text': text,
            'similar': get_similar(text, doc_for_api)
        }


class DocResource(Resource):
    def get(self, doc_id):
        return {
            'id': doc_id,
            'case_number': metadata[doc_id]['case_num'],
            'pdf_path': kad_pdf_path(metadata[doc_id]),
            'text': data_samples[doc_id],
            'similar': get_similar(data_samples[doc_id], doc_for_api, True),
        }


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

app = Flask(__name__)
app.config.from_object('recommenders.webapp_config')

db = SQLAlchemy(app)

from .db import *

api = Api(app)
api.add_resource(UploadResource, '/api/upload')
api.add_resource(DocResource, '/api/doc/<int:doc_id>')

corpus, data_samples, dictionary, metadata = load_uci(app.config['DOCS_LOCATION'])
tfidf = TfidfModel(dictionary=dictionary, smartirs='ntc')
del corpus

logging.info('Unpickling models')
lsi = load(app.config['LSI_PICKLE'])
lda = load(app.config['LDA_PICKLE'])
artm = load(app.config['ARTM_PICKLE'])
artm2 = load(app.config['ARTM2_PICKLE'])
d2v = load(app.config['D2V_PICKLE'])
logging.info('Unpickling finished')


@app.route('/')
def index():
    random_docs = [(idx, data_samples[idx]) for idx in random.sample(range(len(data_samples)), 10)]
    return render_template('index.html', docs=random_docs)


@app.route('/doc/<int:idx>')
def doc(idx):
    similar = get_similar(data_samples[idx], lambda sim: (sim, data_samples[sim]), True)
    return render_template('doc.html', doc=data_samples[idx], case_num=metadata[idx]['case_num'], **similar)


@app.route('/upload', methods=['POST'])
def similar_for_file():
    file = request.files['file']

    tmp_file = tempfile.NamedTemporaryFile()
    file.save(tmp_file)
    text = preprocess(unpack.from_file(tmp_file.name)['content'])
    tmp_file.close()

    similar = get_similar(text, lambda sim: (sim, data_samples[sim]))
    return render_template('doc.html', doc=text, idx=-1, **similar)


@app.route('/rate/<int:doc_id>/<int:rec_id>', methods=['POST'])
def rate_recommendation(doc_id, rec_id):
    score = int(request.values['score'])
    ip = request.remote_addr

    rating = Rating(doc_id=doc_id, recommendation_id=rec_id, value=score, ip=ip)
    db.session.add(rating)
    db.session.commit()


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)
