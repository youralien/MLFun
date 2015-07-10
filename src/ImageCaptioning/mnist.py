import ipdb
import theano.tensor as T
import numpy as np

# # # # # # # # # # #
# DataPreprocessing #
# # # # # # # # # # #
from fuel.datasets import MNIST
from fuel.transformers import (Flatten, SingleMapping, Padding, Mapping,
    FilterSources, Cast, Rename)
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

from foxhound.transforms import SeqPadded

from blocks.bricks import (Initializable)

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
        , iteration_scheme=ShuffledScheme(dataset.num_examples, batch_size=batch_size)))
    stream = Digit2String(stream, which_sources=('targets',))
    stream = Words2Indices(stream, which_sources=('targets',))
    stream = Padding(stream)

    # Padded Words are now in the targets_mask
    stream = Cast(stream, 'uint8', which_sources='targets_mask')
    # stream = FilterSources(stream, sources=("features", "targets_mask"))
    # stream = Rename(stream, names={"features": "features", "targets_mask": "targets"})
    
    return stream

def getTrainStream():
    train = MNIST(('train',))
    return getDataStream(train, batch_size=200)

def getTestStream():
    test = MNIST(('test',))
    return getDataStream(test, batch_size=1000)

# # # # # # # # # # # 
# Modeling Building #
# # # # # # # # # # # 
from abc import ABCMeta, abstractmethod
from six import add_metaclass

from theano import tensor

from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.base import application, Brick
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import GatedRecurrent
from blocks.bricks import Initializable, Linear
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)

theano.config.compute_test_value = 'warn' # Use 'warn' to activate this feature
im = T.matrix('features')
chars = T.matrix('targets')
chars_mask = T.matrix('targets_mask')

im.tag.test_value = np.zeros((2, 28*28))
im.tag.test_value = np.zeros((2, 5))

class MNISTPoet(Initializable):

    def __init__(self, image_dim, dim, alphabet_size=26, **kwargs):

        image_embedding = Linear(
              input_dim=image_dim
            , output_dim=dim
            , name='image_embedding'
            )

        lookup = LookupTable(alphabet_size, dim)

        transition = GatedRecurrent(name='transition', dim=dim)

        readout = Readout(
              readout_dim=alphabet_size
            , source_names=[transition.apply.states[0],
                            attention]
            , emitter=SoftmaxEmitter(name='emitter')
            , feedback_brick=LookupFeedback(num_outputs=alphabet_size, feedback_dim=dim)
            , name="readout"
            )

        generator = SequenceGenerator(
              readout=readout
            , transition=transition
            )
        
        self.image_embedding = image_embedding
        self.lookup = lookup
        self.transition = transition
        self.generator = generator
    
    @application
    def apply(self, image_vects, chars, chars_mask):
        
        image_embedding = self.image_embedding.apply(image_vects)
        
        self.sequence = 
        self.generator.cost_matrix()
        self.lookup.apply(chars)
        return

    @application
    def generate(self, chars):

    
    # @application
    # def cost(self, image_vects, chars, chars_mask):

    #     self.image_embedding.apply()

    #     self.sequence = 
    #     self.generator.cost_matrix()
    #     self.lookup.apply(chars)
    #     return

    # @application
    # def generate(self, chars):


