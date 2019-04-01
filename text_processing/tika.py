import os

from tika import unpack

from text_processing.base import preprocess


def parse(path):
    try:
        parsed = unpack.from_file(path)  # unpack is faster in this case
        return preprocess(parsed['content'])
    except Exception as e:
        print('Exception while reading %s: %s' % (path, e))
        return None


texts = [parse('../out/docs/' + path) for path in os.listdir('../out/docs')]
