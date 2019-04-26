import os

root_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."

DOCS_LOCATION = os.environ.get('DOCS_LOCATION', root_dir + '/out/docs_simple2')
N_SAMPLES = int(os.environ.get('N_SAMPLES', 20000))
N_TOPICS = int(os.environ.get('N_TOPICS', 800))

LSI_PICKLE = os.environ.get('LSI_PICKLE', root_dir + '/out/lsi.pickle')
LDA_PICKLE = os.environ.get('LDA_PICKLE', root_dir + '/out/lda.pickle')
