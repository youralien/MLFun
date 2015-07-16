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
    stream = FilterSources(stream, sources=("features", "targets"))
    # stream = Rename(stream, names={"features": "features", "targets_mask": "targets"})

    return stream

def getTrainStream(batch_size=128):
    train = MNIST(('train',))
    return getDataStream(train, batch_size=batch_size)

def getTestStream(batch_size=1000):
    test = MNIST(('test',))
    return getDataStream(test, batch_size=batch_size)

# # # # # # # # # # #
# Modeling Building #
# # # # # # # # # # #
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.base import application, Brick, lazy
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import LSTM
from blocks.bricks import Initializable, Linear
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, AbstractEmitter, LookupFeedback)
from blocks.monitoring import aggregation
from blocks.graph import ComputationGraph
from modelbuilding import GatedRecurrentWithInitialState

def l2(arr, axis=None):
    """Return the L2 norm of a tensor.
    Parameters
    ----------
    arr : Theano variable.
        The variable to calculate the norm of.
    axis : integer, optional [default: None]
        The sum will be performed along this axis. This makes it possible to
        calculate the norm of many tensors in parallel, given they are organized
        along some axis. If not given, the norm will be computed for the whole
        tensor.
    Returns
    -------
    res : Theano variable.
        If ``axis`` is ``None``, this will be a scalar. Otherwise it will be
        a tensor with one dimension less, where the missing dimension
        corresponds to ``axis``.
    Examples
    --------
    >>> from theano.printing import pprint
    >>> v = T.vector()
    >>> this_norm = l2(v)
    >>> pprint(this_norm)
    'sqrt((Sum((<TensorType(float32, vector)> ** TensorConstant{2})) + TensorConstant{9.99999993923e-09}))'
    >>> m = T.matrix()
    >>> this_norm = l2(m, axis=1)
    >>> pprint(this_norm)
    'sqrt((Sum{1}((<TensorType(float32, matrix)> ** TensorConstant{2})) + TensorConstant{9.99999993923e-09}))'
    >>> m = T.matrix()
    >>> this_norm = l2(m)
    >>> pprint(this_norm)
    'sqrt((Sum((<TensorType(float32, matrix)> ** TensorConstant{2})) + TensorConstant{9.99999993923e-09}))'
    """
    return T.sqrt((arr ** 2).sum(axis=axis) + 1e-8)

class SimilarityEmitter(AbstractEmitter):
    """An emitter for the trivial case when readouts are outputs.
    Parameters
    ----------
    readout_dim : int
        The dimension of the readout.
    Notes
    -----
    By default :meth:`cost` always returns zero tensor.
    """
    @lazy(allocation=['readout_dim'])
    def __init__(self, readout_dim, **kwargs):
        super(SimilarityEmitter, self).__init__(**kwargs)
        self.readout_dim = readout_dim

    @application
    def emit(self, readouts):
        return readouts

    @application
    def cost(self, readouts, outputs):
        l2readouts = l2(readouts)
        l2outputs = l2(outputs)
        readouts = readouts // l2readouts
        outputs = outputs // l2outputs

        # cosine similarity cost.  if the vectors predicted are the actual ones
        # cos_sim will be 1, 1 -1 = 0 which is 0 cost.
        margin = 1 # alpha term, should not be more than 1!

        # pairwise ranking loss (https://github.com/youralien/skip-thoughts/blob/master/eval_rank.py)
        cost = margin - (readouts * outputs).mean(axis=1)
        cost = cost * (cost > 0.) # this is like the max(0, pairwise-ranking-loss)
        cost = cost.mean(0)
        cost.name = "similarity"
        return cost

    @application
    def initial_outputs(self, batch_size):
        return T.zeros((batch_size, self.readout_dim))

    def get_dim(self, name):
        if name == 'outputs':
            return self.readout_dim
        return super(SimilarityEmitter, self).get_dim(name)

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

        transition = GatedRecurrentWithInitialState(
            name='transition', dim=dim
            )

        readout = Readout(
              readout_dim=alphabet_size
            , source_names=["states"]
            , emitter=SoftmaxEmitter(name='emitter')
            , feedback_brick=LookupFeedback(num_outputs=alphabet_size, feedback_dim=dim)
            , name="readout"
            )

        generator = SequenceGenerator(
              readout=readout
            , transition=transition
            )

        self.image_embedding = image_embedding
        self.transition = transition
        self.generator = generator

        self.children = [
                  self.image_embedding
                , self.transition
                , self.generator
                , ]

    @application(inputs=["image_vects", "chars"], outputs=['out'])
    def cost(self, image_vects, chars):
        # shape (batch, features)
        image_embedding = self.image_embedding.apply(image_vects)

        # will the initialize() ruin everything?
        
        cost = aggregation.mean(
              self.generator.cost_matrix(
                chars, cnn_context=image_embedding).sum()
            , chars.shape[1]
            )
        
        # cost = aggregation.mean(cost, chars.shape[1])
        # cost = aggregation.mean(
        #       self.generator.cost_matrix(
        #         chars, cnn_context=image_embedding).sum()
        #     , embedding.shape[1]
        #     )

        # shape (batch, sequence_pad_length + 1, features)
        #chars_mask = T.ones_like(image_embedding)
        #chars_mask = T.concatenate((T.ones_like(image_embedding), chars_mask), axis=1)
        #to_inputs= self.to_inputs.apply(embedding)
        #hidden, cells = self.transition.apply(inputs=to_inputs)
        return cost

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


# theano.config.compute_test_value = 'warn' # Use 'warn' to activate this feature

# Tensors are sensitive as fuck to the dtype, especially in theano.scan
im = T.matrix('features')
chars = T.lmatrix('targets')
#chars_mask = T.matrix('targets_mask')
im.tag.test_value = np.zeros((2, 28*28), dtype='float32')
chars.tag.test_value = np.zeros((2, 5), dtype='int64')

batch_size = 128
image_dim = 784
embedding_dim = 88
mnistpoet = MNISTPoet(
          image_dim=image_dim
        , dim=embedding_dim
        , biases_init=Constant(0.)
        , weights_init=IsotropicGaussian(0.02)
        )
mnistpoet.initialize()
dimchars = chars.dimshuffle(1, 0)
cost = mnistpoet.cost(im, dimchars)

f = theano.function([im, chars], cost)

data = getTrainStream().get_epoch_iterator().next()


costvalue = f(*data)

ipdb.set_trace()

cg = ComputationGraph(cost)

ipdb.set_trace()
# # # # # # # # # # #
# Modeling Training #
# # # # # # # # # # #
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent, AdaDelta
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions import Printing, ProgressBar, FinishAfter

algorithm = GradientDescent(
      cost=cost
    , parameters=cg.parameters
    , step_rule=AdaDelta()
    )
main_loop = MainLoop(
      model=Model(cost)
    , data_stream=getTrainStream()
    , algorithm=algorithm
    , extensions=[
          DataStreamMonitoring(
              [cost]
            , getTrainStream()
            , prefix='train')
        , DataStreamMonitoring(
              [cost]
            , getTestStream()
            , prefix='test')
        , ProgressBar()
        , Printing()
        , FinishAfter(after_n_epochs=15)
        ]
    )
main_loop.run()
