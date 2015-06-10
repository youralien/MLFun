import os
import numpy as np
import pandas as pd
import gzip

def glove(n_words=100000):
    f = gzip.GzipFile('/home/alec/datasets/glove/glove.840B.300d.txt.gz', 'r')
    glove = {}
    for i, row in enumerate(f):
        if i % 10000 == 0: print i
        if i >= n_words: break
        row = row.split(' ')
        word = row[0].lower()
        if word not in glove:
            glove[word] = [float(n) for n in row[1:]]
    return glove

def clean(text):
    return text.strip().replace('<br /><br />', '\n')

def load_imdb_part(data_dir, part_path):
    current_dir = os.path.join(data_dir, part_path, 'pos')
    fs = [os.path.join(current_dir, path) for path in os.listdir(current_dir)]
    current_dir = os.path.join(data_dir, part_path, 'neg')
    fs += [os.path.join(current_dir, path) for path in os.listdir(current_dir)]
    text = [clean(open(f).read()) for f in fs]
    labels = np.asarray([[1] for _ in range(12500)]+[[0] for _ in range(12500)])
    return text, labels.flatten()

def imdb(data_dir=None):
    if data_dir is None:
        import urllib
        import tarfile
        if not os.path.exists('aclImdb/'):
            print 'data_dir not given and data not local - downloading'
            url = 'http://ai.stanford.edu/~amaas/data/sentiment/'
            fname = 'aclImdb_v1.tar.gz'
            urllib.urlretrieve(url+fname, fname)
            f = tarfile.open(fname, 'r:gz')
            f.extractall('.')
            os.remove(fname)
        data_dir = 'aclImdb'
    elif not os.path.exists(data_dir):
        raise IOError('Directory given does not exist')

    trX, trY = load_imdb_part(data_dir, 'train')
    teX, teY = load_imdb_part(data_dir, 'test')
    return trX, teX, trY, teY