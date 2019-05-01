import logging
import random
import tempfile

from flask import Flask, render_template, request
from flask_restful import Api, Resource
from gensim.models import TfidfModel
from tika import unpack

from recommenders.models import LsiModel, LdaModel, Doc2vecModel
from recommenders.util import load_uci, load_model, tokenize_tfidf
from text_processing.base import preprocess


def doc_for_api(doc_id):
    return {'id': int(doc_id), 'url': api.url_for(DocResource, doc_id=doc_id, _external=True)}


def similar_for_idx(idx):
    return {
        'id': idx,
        'text': data_samples[idx],
        'similar_lsi': [doc_for_api(sim) for sim in lsi.get_similar(corpus[idx])[1:]],
        'similar_lda': [doc_for_api(sim) for sim in lda.get_similar(corpus[idx])[1:]],
        'similar_d2v': [doc_for_api(sim) for sim in lda.get_similar(corpus[idx])[1:]],
    }


class UploadResource(Resource):
    def post(self):
        file = request.files['file']

        tmp_file = tempfile.NamedTemporaryFile()
        file.save(tmp_file)
        text = preprocess(unpack.from_file(tmp_file.name)['content'])
        tmp_file.close()

        doc = tokenize_tfidf(text, tfidf, dictionary)

        return {
            'similar_lsi': [doc_for_api(sim) for sim in lsi.get_similar(doc)],
            'similar_lda': [doc_for_api(sim) for sim in lda.get_similar(doc)],
            'similar_d2v': [doc_for_api(sim) for sim in d2v.get_similar(text)],
        }


class DocResource(Resource):
    def get(self, doc_id):
        return {
            'id': doc_id,
            'text': data_samples[doc_id],
            'similar_lsi': [doc_for_api(sim) for sim in lsi.get_similar(corpus[doc_id])[1:]],
            'similar_lda': [doc_for_api(sim) for sim in lda.get_similar(corpus[doc_id])[1:]],
            'similar_d2v': [doc_for_api(sim) for sim in d2v.get_similar(data_samples[doc_id])[1:]],
        }


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

app = Flask(__name__)
app.config.from_object('recommenders.webapp_config')

api = Api(app)
api.add_resource(UploadResource, '/api/upload')
api.add_resource(DocResource, '/api/doc/<int:doc_id>')

corpus, data_samples = load_uci(app.config['DOCS_LOCATION'])
dictionary = corpus.create_dictionary()
tfidf = TfidfModel(dictionary=dictionary, smartirs='ntc')
corpus = [tfidf[doc] for doc in corpus]

lsi = load_model(lambda: LsiModel(corpus, dictionary, app.config['N_TOPICS']), app.config['LSI_PICKLE'])
lda = load_model(lambda: LdaModel(corpus, dictionary, app.config['N_TOPICS']), app.config['LDA_PICKLE'])
# artm = load_model(lambda: BigArtmModel(app.config['UCI_FOLDER'], dictionary, app.config['N_TOPICS']))
d2v = load_model(lambda: Doc2vecModel(data_samples, app.config['N_TOPICS']), app.config['D2V_PICKLE'])

# cm = CoherenceModel(model=lsi.lsi, dictionary=dictionary, corpus=corpus, coherence='u_mass')
# print(cm.compare_models([lsi.lsi, lda.lda]))


@app.route('/')
def index():
    random_docs = [(idx, data_samples[idx]) for idx in random.sample(range(len(corpus)), 10)]
    return render_template('index.html', docs=random_docs)


@app.route('/doc/<int:idx>')
def doc(idx):
    similar_lsi = [(sim, data_samples[sim]) for sim in lsi.get_similar(corpus[idx])[1:]]
    similar_lda = [(sim, data_samples[sim]) for sim in lda.get_similar(corpus[idx])[1:]]
    similar_d2v = [(sim, data_samples[sim]) for sim in d2v.get_similar(data_samples[idx])[1:]]
    return render_template('doc.html', doc=data_samples[idx], idx=idx,
                           similar_lsi=similar_lsi, similar_lda=similar_lda, similar_d2v=similar_d2v)


@app.route('/upload', methods=['POST'])
def similar_for_file():
    file = request.files['file']

    tmp_file = tempfile.NamedTemporaryFile()
    file.save(tmp_file)
    text = preprocess(unpack.from_file(tmp_file.name)['content'])
    tmp_file.close()

    doc = tokenize_tfidf(text, tfidf, dictionary)

    similar_lsi = [(sim, data_samples[sim]) for sim in lsi.get_similar(doc)]
    similar_lda = [(sim, data_samples[sim]) for sim in lda.get_similar(doc)]
    similar_d2v = [(sim, data_samples[sim]) for sim in d2v.get_similar(text)]
    return render_template('doc.html', doc=text, idx=-1,
                           similar_lsi=similar_lsi, similar_lda=similar_lda, similar_d2v=similar_d2v)


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)
