import itertools
import os
import pickle
import random
import tempfile
from time import time

from flask import Flask, render_template, request
from tika import unpack

from recommenders.lsi import LsiModel
from text_processing.base import preprocess
from text_processing.simple import parse_all

app = Flask(__name__)
app.config.from_object('recommenders.webapp_config')

print("Loading the corpus...")
t0 = time()
data_samples = list(itertools.islice(parse_all(app.config['DOCS_LOCATION'], from_cache=True), app.config['N_SAMPLES']))
print("loaded %d samples in %0.3fs." % (len(data_samples), time() - t0))

if app.config['LSI_PICKLE'] and os.path.exists(app.config['LSI_PICKLE']):
    with open(app.config['LSI_PICKLE'], 'rb') as f:
        model = pickle.load(f)
else:
    model = LsiModel(data_samples, app.config['N_TOPICS'])
    if app.config['LSI_PICKLE']:
        with open(app.config['LSI_PICKLE'], 'wb') as f:
            pickle.dump(model, f)


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
