import os
import csv
import ipdb
import operator

import numpy as np
import pandas as pd
import theano

from fuel import config
from fuel.transformers import Transformer
from foxhound.utils import shuffle
from pycocotools.coco import COCO

dataDir='/home/luke/datasets/coco'

def coco(mode="dev", batch_size=64, n_captions=1):

    # train_fns
    dataType='train2014'
    train_fns = os.listdir("%s/features/%s"%(dataDir, dataType))

    # reduce it to a dev set
    if mode == "dev":
        train_fns = shuffle(train_fns)[:batch_size*50]
    trX, trY = loadFeaturesTargets(train_fns, dataType, n_captions)

    # val_fns
    dataType='val2014'
    test_fns = os.listdir("%s/features/%s"%(dataDir, dataType))

    # reduce it to a dev set
    if mode == "dev":
        test_fns = shuffle(test_fns)[:batch_size*25]
    teX, teY = loadFeaturesTargets(test_fns, dataType, n_captions)

    return trX, teX, trY, teY

def sbuXYFilenames(n_examples):
    """
    n_examples to try to load.  It might not load all of them
    """
    sbu_path = "/home/luke/datasets/sbu/"
    sbu_feature_path = os.path.join(sbu_path, "features")
    sbu_caption_path = os.path.join(sbu_path, "SBU_captioned_photo_dataset_captions.txt")
    sbu_urls_path = os.path.join(sbu_path, "SBU_captioned_photo_dataset_urls.txt")
    fns = os.listdir(sbu_feature_path)[:n_examples]
    
    print "Reading SBU captions"
    f = open(sbu_caption_path, 'rb')
    captions = f.read().splitlines()
    f.close()

    X, Y = [], []
    successes = []

    print "Loading in SBU Features"
    for i in range(len(fns)):
        fn = fns[i]
        try:
            # fn should be SBU_%d
            index = int(fn[4:].split(".")[0])
            X.append(np.load(os.path.join(sbu_feature_path, fn)))
            Y.append([captions[index]])
            successes.append(i)
        except:
            continue

    # get only the successful fns
    fns = operator.itemgetter(*successes)(fns)
    print "SBU Done!"

    return X, Y, fns

def cocoXYFilenames(n_captions=5, dataType='val2014'):
    """Helps when you are evaluating and want the filenames
    associated with the features and target variables

    Parameters
    ----------
    n_captions: integer
        how many captions to load for the image

    dataType: 'val2014' or 'train2014'

    Returns
    -------
    X: the features
    Y: the targets
    filenames: the filenames corresponding to each
    """
    fns = os.listdir("%s/features/%s"%(dataDir, dataType))
    fns = shuffle(fns)
    X, Y = loadFeaturesTargets(fns, dataType, n_captions)

    return X, Y, fns

def loadFeaturesTargets(fns, dataType, n_captions=1):
    """
    Note: filenames should come from the same type of dataType.

    filenames from val2014, for example, should have dataType val2014
    Parameters
    ----------
    fns: filenames, strings

    dataType: string folder, i.e. train2014, val2014

    n_captions: int, number of captions for each image to load

    Returns
    -------
    X: list of im_vects
        1st list length = len(fns)
        vectors are shape (4096, )

    Y: list of list of captions.
        1st list length = len(fns)
        sublist length = n_captions
    """
    annFile = '%s/annotations/captions_%s.json'%(dataDir,dataType)
    caps=COCO(annFile)

    X = []
    Y = []

    for fn in fns:
        # Features
        x = np.load('%s/features/%s/%s'%(dataDir, dataType, fn))

        # Targets
        annIds = caps.getAnnIds(imgIds=getImageId(fn));
        anns = caps.loadAnns(annIds)

        # sample n_captions per image
        anns = shuffle(anns)
        captions = [getCaption(anns[i]) for i in range(n_captions)]

        X.append(x)
        Y.append(captions)

    return X, Y

def npy2jpg(fn):
    name, ext = fn.split(".")
    return name + ".jpg"

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

def fillOutFilenames(filenames, n_captions):
    new_fns = []
    for fn in filenames:
        new_fns.extend([fn for i in range(n_captions)])
    return new_fns

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
        raw = pd.read_csv(data_path, header=None, sep=' ', quoting=csv.QUOTE_NONE, nrows=50000)
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
    # trX, teX, trY, teY = coco(mode="dev")
    # ipdb.set_trace()


    def test_fillOutFilenames(n_captions=3):
        dataDir='/home/luke/datasets/coco'
        dataType='val2014'
        test_fns = os.listdir("%s/features/%s"%(dataDir, dataType))
        test_fns = test_fns[:128]

        new_fns = fillOutFilenames(test_fns, n_captions)
        print new_fns

    test_fillOutFilenames()
    ipdb.set_trace()
