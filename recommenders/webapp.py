import random
import tempfile

from flask import Flask, render_template, request
from flask_restful import Api, Resource
from gensim.models import CoherenceModel
from tika import unpack

from recommenders.util import load_lsi, load_lda, load_uci
from text_processing.base import preprocess


def doc_for_api(doc_id):
    return {'id': int(doc_id), 'url': api.url_for(DocResource, doc_id=doc_id, _external=True)}


class UploadResource(Resource):
    def post(self):
        file = request.files['file']

        tmp_file = tempfile.NamedTemporaryFile()
        file.save(tmp_file)
        text = preprocess(unpack.from_file(tmp_file.name)['content'])
        tmp_file.close()

        return {
            'similar_lsi': [doc_for_api(sim) for sim in lsi.get_similar_for_text(text)[:10]],
            'similar_lda': [doc_for_api(sim) for sim in lda.get_similar_for_text(text)[:10]],
        }


class DocResource(Resource):
    def get(self, doc_id):
        return {
            'id': doc_id,
            'text': data_samples[doc_id],
            'similar_lsi': [doc_for_api(sim) for sim in lsi.get_similar_ids(doc_id)[:10]],
            'similar_lda': [doc_for_api(sim) for sim in lda.get_similar_ids(doc_id)[:10]],
        }


app = Flask(__name__)
app.config.from_object('recommenders.webapp_config')

api = Api(app)
api.add_resource(UploadResource, '/api/upload')
api.add_resource(DocResource, '/api/doc/<int:doc_id>')

corpus, data_samples = load_uci(app.config['DOCS_LOCATION'])
dictionary = corpus.create_dictionary()
lsi = load_lsi(corpus, dictionary, app.config['N_TOPICS'])
lda = load_lda(corpus, dictionary, app.config['N_TOPICS'])

cm = CoherenceModel(model=lsi.lsi, dictionary=corpus.dictionary, corpus=corpus, coherence='u_mass')
print(cm.compare_models([lsi.lsi, lda.lda]))


@app.route('/')
def index():
    random_docs = [(idx, data_samples[idx]) for idx in random.sample(range(len(corpus)), 10)]
    return render_template('index.html', docs=random_docs)


@app.route('/doc/<int:idx>')
def doc(idx):
    similar_lsi = [(sim, data_samples[sim]) for sim in lsi.get_similar_ids(idx)[:10]]
    similar_lda = [(sim, data_samples[sim]) for sim in lda.get_similar_ids(idx)[:10]]
    return render_template('doc.html', doc=data_samples[idx], idx=idx, similar_lsi=similar_lsi, similar_lda=similar_lda)


@app.route('/upload', methods=['POST'])
def similar_for_file():
    file = request.files['file']

    tmp_file = tempfile.NamedTemporaryFile()
    file.save(tmp_file)
    text = preprocess(unpack.from_file(tmp_file.name)['content'])
    tmp_file.close()

    similar_lsi = [(sim, data_samples[sim]) for sim in lsi.get_similar_for_text(text)[:10]]
    similar_lda = [(sim, data_samples[sim]) for sim in lda.get_similar_for_text(text)[:10]]
    return render_template('doc.html', doc=text, idx=-1, similar_lsi=similar_lsi, similar_lda=similar_lda)


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)
