import os

root_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."

UCI_FOLDER = os.environ.get('DOCS_LOCATION', root_dir + '/out')
DOCS_LOCATION = os.environ.get('DOCS_LOCATION', UCI_FOLDER + '/corpus.uci')
N_SAMPLES = int(os.environ.get('N_SAMPLES', 20000))
N_TOPICS = int(os.environ.get('N_TOPICS', 800))

LSI_PICKLE = os.environ.get('LSI_PICKLE', root_dir + '/out/lsi.pickle')
LDA_PICKLE = os.environ.get('LDA_PICKLE', root_dir + '/out/lda.pickle')
D2V_PICKLE = os.environ.get('D2V_PICKLE', root_dir + '/out/d2v.pickle')
