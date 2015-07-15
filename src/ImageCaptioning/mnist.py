import ipdb
import theano
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


class Words2Indices(SingleMapping):

    # Dictionaries
    all_chars = ([chr(ord('a') + i + 1) for i in range(26)]) # alphabeti only
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
    # stream = Cast(stream, 'int32', which_sources=('targets', 'targets_mask'))
    # stream = Cast(stream, 'float8', which_sources=('features',))
    # stream = FilterSources(stream, sources=("features", "targets_mask"))
    # stream = Rename(stream, names={"features": "features", "targets_mask": "targets"})

    return stream

def getTrainStream(batch_size=200):
    train = MNIST(('train',))
    return getDataStream(train, batch_size=batch_size)

def getTestStream(batch_size=1000):
    test = MNIST(('test',))
    return getDataStream(test, batch_size=batch_size)

# # # # # # # # # # #
# Modeling Building #
# # # # # # # # # # #
from abc import ABCMeta, abstractmethod
from six import add_metaclass

from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.base import application, Brick
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import LSTM
from blocks.bricks import Initializable, Linear
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)


class MNISTPoet(Initializable):

    def __init__(self, image_dim, dim, alphabet_size=26, **kwargs):
        super(MNISTPoet, self).__init__(**kwargs)

        # make image dimension of the embedding, so we can initialize
        # the hidden state with it
        image_embedding = Linear(
              input_dim=image_dim
            , output_dim=dim
            , name='image_embedding'
            )

        lookup = LookupTable(alphabet_size, dim)
        to_inputs = Linear(
              input_dim=dim
            , output_dim=dim*4
            , name="to_inputs"
            )

        transition = LSTM(name='transition', dim=dim)

#        readout = Readout(
#              readout_dim=alphabet_size
#            , source_names=[image_embedding.apply.outputs[0],
#                            transition.apply.states[0]]
#            , emitter=SoftmaxEmitter(name='emitter')
#            , feedback_brick=LookupFeedback(num_outputs=alphabet_size, feedback_dim=dim)
#            , name="readout"
#            )
#
#        generator = SequenceGenerator(
#              readout=readout
#            , transition=transition
#            )

        self.image_embedding = image_embedding
        self.lookup = lookup
        self.to_inputs = to_inputs
        self.transition = transition

        self.children = [
                  self.image_embedding
                , self.lookup
                , self.to_inputs
                , self.transition
                ]

    @application(inputs=["image_vects", "chars"], outputs=['out'])
    def apply(self, image_vects, chars):
        # shape (batch, features)
        image_embedding = self.image_embedding.apply(image_vects)

        # shape (batch, 1, features)
        image_embedding = image_embedding.dimshuffle(0, 'x', 1)

        # shape (batch, sequence_pad_length, features)
        text_embedding = self.lookup.apply(chars)

        # shape (batch, sequence_pad_length + 1, features)
        embedding = T.concatenate((image_embedding, text_embedding), axis=1)

        # shape (batch, sequence_pad_length + 1, features)
        #chars_mask = T.ones_like(image_embedding)
        #chars_mask = T.concatenate((T.ones_like(image_embedding), chars_mask), axis=1)
        to_inputs= self.to_inputs.apply(embedding)
        hidden, cells = self.transition.apply(inputs=to_inputs)
        return hidden

    @application
    def generate(self, chars):
        pass

    # @application
    # def cost(self, image_vects, chars, chars_mask):

    #     self.image_embedding.apply()

    #     self.sequence = 
    #     self.generator.cost_matrix()
    #     self.lookup.apply(chars)
    #     return

    # @application
    # def generate(self, chars):


theano.config.compute_test_value = 'off' # Use 'warn' to activate this feature

# Tensors are sensitive as fuck to the dtype, especially in theano.scan
im = T.matrix('features')
chars = T.lmatrix('targets')
#chars_mask = T.matrix('targets_mask')
# im.tag.test_value = np.zeros((2, 28*28))
# chars.tag.test_value = np.zeros((2, 5))

batch_size = 128
image_dim = 784
embedding_dim = 100
mnistpoet = MNISTPoet(
          image_dim=image_dim
        , dim=embedding_dim
        , biases_init=Constant(0.)
        , weights_init=IsotropicGaussian(0.02)
        )
mnistpoet.initialize()
output = mnistpoet.apply(im, chars)

f = theano.function([im, chars], output)

data = getTrainStream().get_epoch_iterator().next()


emb = f(data[0], data[2])
ipdb.set_trace()
