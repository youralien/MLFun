import os
import csv
import ipdb

import numpy as np
import pandas as pd
import theano

from fuel import config
from fuel.transformers import Transformer
from foxhound.utils import shuffle
from pycocotools.coco import COCO

dataDir='/home/luke/datasets/coco'

def coco(mode="dev"):

    # train_fns
    dataType='train2014'
    train_fns = os.listdir("%s/features/%s"%(dataDir, dataType))

    # reduce it to a dev set
    if mode == "dev":
        train_fns = train_fns[:128*100]
    trX, trY = loadFeaturesTargets(train_fns, dataType)
    
    # val_fns
    dataType='val2014'
    test_fns = os.listdir("%s/features/%s"%(dataDir, dataType))

    # reduce it to a dev set
    if mode == "dev":
        test_fns = test_fns[:128]
    teX, teY = loadFeaturesTargets(test_fns, dataType)

    return trX, teX, trY, teY 

def loadFeaturesTargets(fns, dataType):
    """
    Note: filenames should come from the same type of dataType.

    filenames from val2014, for example, should have dataType val2014
    Parameters
    ----------
    fns: filenames, strings


    """
    annFile = '%s/annotations/captions_%s.json'%(dataDir,dataType)
    caps=COCO(annFile)
    
    X = []
    Y = []

    for fn in fns:
        # Features
        x = np.load('%s/features/%s/%s'%(dataDir, dataType, fn))
        X.append(x)
        
        # Targets
        annIds = caps.getAnnIds(imgIds=getImageId(fn));
        anns = caps.loadAnns(annIds)

        # Just get one (the first) caption for now...
        Y.append(getCaption(anns[0]))

    return X, Y

def getImageId(fn):
    """Filename to image id

    Parameters
    ----------
    fn: a string
        filename of the COCO dataset.

        example:
        COCO_val2014_000000581929.npy

    Returns
    imageId: an int
    """
    return int(fn.split("_")[-1].split('.')[0])

def getCaption(ann):
    """gets Caption from the COCO annotation object
    
    Parameters
    ----------
    ann: list of annotation objects
    """
    return str(ann["caption"])

# Foxhound + Fuel
class FoxyDataStream(object):
    """FoxyDataStream attempts to merge the gap between fuel DataStreams and
    Foxhound iterators.

    The place we will be doing this merge is in the blocks MainLoop. Inserting
    a FoxyDataStream() in place of a DataStream.default_stream()
    will suffice.

    (Note)
    These are broken down into the following common areas
    - dataset which has (features, targets) or (X, Y)
    - iteration_scheme (sequential vs shuffling, batch_size)
    - transforms

    Parameters
    ----------
    data: tuple of X, Y

    sources: tuple of sourcenameX, sourcenameY

    iterator: a Foxhound iterator.  The use is jank right now, but always use
        trXt and trYt as the X and Y transforms respectively
    """

    def __init__(self, data, sources, make_iterator, iteration_scheme=None):
        self.data = data
        self.sources = sources
        self.iterator = make_iterator
        # self.iterator_prototype = make_iterator
        self.iteration_scheme = iteration_scheme # Compatibility with the blocks mainloop

    def get_epoch_iterator(self, as_dict=False):

        # iterator = self.iterator_prototype(None)
        # print iterator
        for datamb in self.iterator.iterXY(*self.data):
            yield dict(zip(self.sources, datamb)) if as_dict else datamb

class FoxyIterationScheme(object):
    """mimics like a Fox a fuel iteration scheme

    Important Attributes
    --------------------
    num_batches: int

    OR

    num_examples: int

    batch_size: int
    """
    def __init__(self, examples, batch_size):
        self.num_examples = examples
        self.batch_size = batch_size

class GloveTransformer(Transformer):
    glove_folder = "glove"
    vector_dim = 0

    def __init__(self, glove_file, data_stream, vectorizer):
        super(GloveTransformer, self).__init__(data_stream)
        dir_path = os.path.join(config.data_path, self.glove_folder)
        data_path = os.path.join(dir_path, glove_file)
        raw = pd.read_csv(data_path, header=None, sep=' ', quoting=csv.QUOTE_NONE, nrows=20000)
        #raw = pd.read_csv(data_path, nrows=400, header=None, sep=' ', quoting=csv.QUOTE_NONE)
        keys = raw[0].values
        self.vectors = raw[range(1, len(raw.columns))].values.astype(theano.config.floatX)
        self.vector_dim = self.vectors.shape[1]
        
        # lookup will have (key, val) -> (word-string, row index in self.vectors)
        row_indexes = range(self.vectors.shape[0])
        self.lookup = dict(zip(keys, row_indexes))
        self.reverse_lookup = dict(zip(row_indexes, keys))
        self.vectorizer = vectorizer

    def get_data(self, request=None):
        if request is not None:
            raise ValueError

        """
        # If the stream is composed of image_reps, strings
        image_reps, strings = next(self.child_epoch_iterator)
        strings = np.vectorize(str.lower)(strings)

        def process_string(s):
            tokens = tokenize(s)

            output = np.zeros((len(tokens), self.vector_dim), dtype=theano.config.floatX)
            for i,t in enumerate(tokens):
                if t in self.lookup:
                    output[i, :] = self.vectors[self.lookup[t]]
            return output

        word_reps = [process_string(s) for s in strings]
        return image_reps, word_reps
        """

        # If the stream is composed of image_reps, codes
        image_reps, codes = next(self.child_epoch_iterator)

        def process_tokens(tokens):
            output = np.zeros((len(tokens), self.vector_dim), dtype=theano.config.floatX)
            for i,t in enumerate(tokens):
                word = self.vectorizer.decoder[t] 
                if word in self.lookup:
                    output[i, :] = self.vectors[self.lookup[word]]
                # else t is UNK or PAD so we leave the output alone (zero padded)
            return output

        word_reps = np.asarray(
              [process_tokens(tokens) for tokens in codes.T]
            , dtype=theano.config.floatX)

        return image_reps, word_reps

class ShuffleBatch(Transformer):
    """Shuffle the Batch, helpful when generating contrastive examples"""
    def __init__(self, data_stream):
        super(ShuffleBatch, self).__init__(data_stream)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = next(self.child_epoch_iterator)
        return shuffle(*data)

if __name__ == '__main__':
    trX, teX, trY, teY = coco(mode="dev")
    ipdb.set_trace()
