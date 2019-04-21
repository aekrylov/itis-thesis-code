import itertools
import tempfile
from time import time

from flask import Flask, render_template, request
from tika import unpack

from recommenders.lsi import LsiModel
from text_processing.base import preprocess
from text_processing.simple import parse_all

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

n_samples = 20000
print("Loading the corpus...")
t0 = time()
data_samples = list(itertools.islice(parse_all("../out/docs_simple2", from_cache=True), n_samples))
print("loaded %d samples in %0.3fs." % (len(data_samples), time() - t0))
model = LsiModel(data_samples, 800)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/doc/<idx>')
def doc(idx):
    idx = int(idx)
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
    app.run()
