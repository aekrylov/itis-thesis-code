import random
import tempfile

from flask import Flask, render_template, request, url_for
from flask_restful import Api, Resource
from tika import unpack

from recommenders.util import load_corpus, load_lsi
from text_processing.base import preprocess


class UploadResource(Resource):
    def post(self):
        file = request.files['file']

        tmp_file = tempfile.NamedTemporaryFile()
        file.save(tmp_file)
        text = preprocess(unpack.from_file(tmp_file.name)['content'])
        tmp_file.close()

        return [{'id': sim, 'url': api.url_for(DocResource, doc_id=sim)} for sim in model.get_similar_for_text(text)[:10]]


class DocResource(Resource):
    def get(self, doc_id):
        similar = [{'id': sim, 'url': api.url_for(DocResource, doc_id=sim)} for sim in model.get_similar_ids(doc_id)[:10]]
        return {
            'id': doc_id,
            'text': data_samples[doc_id],
            'similar': similar
        }


app = Flask(__name__)
app.config.from_object('recommenders.webapp_config')

api = Api(app)
api.add_resource(UploadResource, '/api/upload')
api.add_resource(DocResource, '/api/doc/<int:doc_id>')

data_samples = load_corpus(app.config['DOCS_LOCATION'], app.config['N_SAMPLES'])
model = load_lsi(data_samples, app.config['N_TOPICS'], app.config['LSI_PICKLE'])


@app.route('/')
def index():
    random_docs = [(idx, data_samples[idx]) for idx in random.sample(range(len(data_samples)), 10)]
    return render_template('index.html', docs=random_docs)


@app.route('/doc/<int:idx>')
def doc(idx):
    similar = [(sim, data_samples[sim]) for sim in model.get_similar_ids(idx)[:10]]
    return render_template('doc.html', doc=data_samples[idx], idx=idx, similar=similar)


@app.route('/upload', methods=['POST'])
def similar_for_file():
    file = request.files['file']

    tmp_file = tempfile.NamedTemporaryFile()
    file.save(tmp_file)
    text = preprocess(unpack.from_file(tmp_file.name)['content'])
    tmp_file.close()

    similar = [(sim, data_samples[sim]) for sim in model.get_similar_for_text(text)[:10]]
    return render_template('doc.html', doc=text, idx=-1, similar=similar)


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)
