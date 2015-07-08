import ipdb
import theano.tensor as T
import numpy as np

from fuel.datasets import MNIST
from fuel.transformers import (Flatten, SingleMapping, Padding, Mapping,
    FilterSources, Cast, Rename)
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from foxhound.transforms import SeqPadded


class Words2Indices(SingleMapping):

    # Dictionaries
    all_chars = ([chr(ord('a') + i) for i in range(26)]) # alphabeti only
    code2char = dict(enumerate(all_chars))
    char2code = {v: k for k, v in code2char.items()}

    def __init__(self, data_stream, **kwargs):
        super(Words2Indices, self).__init__(data_stream, **kwargs)

    def words2indices(self, word):
        indices = [self.char2code[char] for char in word]
        return np.asarray(indices, dtype='int')

    def mapping(self, source):
        return map(self.words2indices, source)

class Digit2String(SingleMapping):
    
    digit_strings = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    
    def __init__(self, data_stream, **kwargs):
        super(Digit2String, self).__init__(data_stream, **kwargs)

    def digit2string(self, digit):
        return self.digit_strings[digit]
    
    def mapping(self, source):
        return map(self.digit2string, source)

def getDataStream(dataset, batch_size):
    stream = Flatten(DataStream.default_stream(
          dataset=dataset
        , iteration_scheme=SequentialScheme(dataset.num_examples, batch_size=batch_size)))
    stream = Digit2String(stream, which_sources=('targets',))
    stream = Words2Indices(stream, which_sources=('targets',))
    stream = Padding(stream)

    # Padded Words are now in the targets_mask
    stream = Cast(stream, 'uint8', which_sources='targets_mask')
    stream = FilterSources(stream, sources=("features", "targets_mask"))
    stream = Rename(stream, names={"features": "features", "targets_mask": "targets"})
    
    return stream

def getTrainStream():
    train = MNIST(('train',))
    return getDataStream(train, batch_size=200)

def getTestStream():
    test = MNIST(('test',))
    return getDataStream(test, batch_size=1000)

# features = T.matrix('features')
# targets = 

test_stream = getTestStream()

data = test_stream.get_epoch_iterator().next()
print [len(d) for d in data]