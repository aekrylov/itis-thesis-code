import os

root_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."

UCI_FOLDER = os.environ.get('UCI_FOLDER', root_dir + '/out')
DOCS_LOCATION = os.environ.get('DOCS_LOCATION', UCI_FOLDER + '/corpus.uci')
N_TOPICS = int(os.environ.get('N_TOPICS', 800))

LSI_PICKLE = os.environ.get('LSI_PICKLE', UCI_FOLDER + '/lsi.pickle')
LDA_PICKLE = os.environ.get('LDA_PICKLE', UCI_FOLDER + '/lda.pickle')
D2V_PICKLE = os.environ.get('D2V_PICKLE', UCI_FOLDER + '/d2v.pickle')
ARTM_PICKLE = os.environ.get('ARTM_PICKLE', UCI_FOLDER + '/artm_new.pkl')

RESTFUL_JSON = {
    'ensure_ascii': False,
    'indent': 4
}

SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
SQLALCHEMY_TRACK_MODIFICATIONS = False
