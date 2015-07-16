import ipdb
import operator
import logging
import cPickle as pkl

import theano
import theano.tensor as T
import numpy as np
from picklable_itertools.extras import equizip

from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent, AdaDelta
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions import Printing, ProgressBar, FinishAfter
from blocks.extensions.saveload import Checkpoint
from blocks.serialization import load_parameter_values
from blocks.filter import VariableFilter

from blocks.search import BeamSearch

logger = logging.getLogger(__name__)

# # # # # # # # # # #
# DataPreprocessing #
# # # # # # # # # # #
from fuel.datasets import MNIST
from fuel.transformers import (Flatten, SingleMapping, Padding, Mapping,
    FilterSources, Cast, Rename)
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

from foxhound.transforms import SeqPadded


# Dictionaries
all_chars = ([chr(ord('a') + i + 1) for i in range(26)]) # alphabeti only
code2char = dict(enumerate(all_chars))
# stop character
code2char[0] = "."
char2code = {v: k for k, v in code2char.items()}

class Words2Indices(SingleMapping):

    def __init__(self, data_stream, **kwargs):
        super(Words2Indices, self).__init__(data_stream, **kwargs)

    def words2indices(self, word):
        indices = [char2code[char] for char in word]
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

def getTrainStream(batch_size=200):
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

        cost = aggregation.mean(
              self.generator.cost_matrix(
                chars, cnn_context=image_embedding).sum()
            , chars.shape[1]
            )
        return cost

    @application
    def generate(self, image_vects):
        # shape (batch, features)
        image_embedding = self.image_embedding.apply(image_vects)
        return self.generator.generate(
                  n_steps=5
                , batch_size=image_embedding.shape[0]
                , iterate=True
                , cnn_context=image_embedding
                )

def main(mode, save_path, num_batches=300):

    image_dim = 784
    embedding_dim = 300
    mnistpoet = MNISTPoet(
              image_dim=image_dim
            , dim=embedding_dim
            , biases_init=Constant(0.)
            , weights_init=IsotropicGaussian(0.02)
            )

    if mode == "train":

        # theano.config.compute_test_value = 'warn' # Use 'warn' to activate this feature

        # Tensors are sensitive as fuck to the dtype, especially in theano.scan
        im = T.matrix('features')
        chars = T.lmatrix('targets')
        im.tag.test_value = np.zeros((2, 28*28), dtype='float32')
        chars.tag.test_value = np.zeros((2, 5), dtype='int64')

        mnistpoet.initialize()

        # dimchars is shape (sequences, batches)
        dimchars = chars.dimshuffle(1, 0)
        cost = mnistpoet.cost(im, dimchars)
        cost.name = "sequence_log_likelihood"

        cg = ComputationGraph(cost)

        # # # # # # # # # # #
        # Modeling Training #
        # # # # # # # # # # #
        model = Model(cost)

        algorithm = GradientDescent(
              cost=cost
            , parameters=cg.parameters
            , step_rule=AdaDelta()
            )
        main_loop = MainLoop(
              model=model
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
                #, Checkpoint(save_path, every_n_batches=500, save_separately=["model", "log"])
                #, Checkpoint(save_path, every_n_batches=500)
                , ProgressBar()
                , Printing()
                , FinishAfter(after_n_batches=num_batches)
                ]
            )
        main_loop.run()

        # Training is done; save the generator function with the learned parameters
        sample = mnistpoet.generate(im)
        #f_gen = theano.function([im], sample)
        f_gen = ComputationGraph(sample).get_theano_function()
        ep = getTestStream(batch_size=1).get_epoch_iterator()
        while True:
            im_vects, txt_enc = ep.next()
            mnist_txt = "".join(code2char[code] for code in txt_enc[0])
            print "\nTrying for: ", mnist_txt
            message=("Number Tries to generate correct text?")
            batch_size = int(input(message))
            states, outputs, costs = f_gen(
                    np.repeat(im_vects, batch_size, 0)
                    )
            outputs = list(outputs.T)
            costs = list(costs.T)
            for i in range(len(outputs)):
                outputs[i] = list(outputs[i])
                try:
                    # 0 was my stop character for MNIST alphabetic
                    true_length = outputs[i].index(0)
                except ValueError:
                    # full sequence length
                    true_length = len(outputs[i])
                outputs[i] = outputs[i][:true_length]
                costs[i] = costs[i][:true_length].sum()
            messages = []
            for sample, cost in equizip(outputs, costs):
                message = "({0:0.3f}) ".format(cost)
                message += "".join(code2char[code] for code in sample)
                messages.append((cost, message))
            messages.sort(key=operator.itemgetter(0), reverse=True)
            for _, message in messages:
                print(message)
        ModelIO.save(f_gen, 'predict-mnist/f_gen')
#    elif mode == "sample" or mode == "beam_search" or mode == "generate":
#        im = T.matrix('features')
#        #mnistpoet.initialize()
#        generated = mnistpoet.generate(im)
#        model = Model(generated)
#        logger.info("Loading the Model...")
#        pkl.load(open(save_path), 'rb')
#        #model.set_parameter_values(load_parameter_values(save_path))
#
#        def generate(input_):
#            """Generate ouptut sequences for a given image
#
#            Returns
#            -------
#            outputs: list of lists
#                Trimmed output sequences
#            costs: list
#                The negative log-likelihood of generating the respective
#                sequences.
#            """
#            if mode == "beam_search":
#                print "No beamsearch available"
#                #samples, = VariableFilter(
#                #    bricks=[mnistpoet.generator], name="outputs")(
#                #        ComputationGraph(generated[1]))
#                #beam_search = BeamSearch(samples)
#                ## confused if chars: should be input_ or input_ should image
#                #outputs, costs = beam_search.search(
#                #    {chars: input_i
#            else:
#                import ipdb; ipdb.set_trace();
#                _1, outputs, _2, _3, costs = (
#                    model.get_theano_function()(input_))
#                outputs = list(outputs.T)
#                costs = list(costs.T)
#                for i in range(len(outputs)):
#                    outputs[i] = list(outputs[i])
#                    try:
#                        # 0 was my stop character for MNIST alphabetic characters
#                        true_length = outputs[i].index(0)
#                    except ValueError:
#                        true_length = len(outputs[i])
#                    # true length helps me 'mask' stop characters in the cost calc
#                    outputs[i] = outputs[i][:true_length]
#                    costs[i] = costs[i][:true_length].sum()
#            return outputs, costs
#        #main_loop = pkl.load(open(save_path, 'rb'))
#        #generator = main_loop.model
#        f_gen = ModelIO.load('predict-mnist/f_gen')
#        # get one example at a time to evaluate
#        ep = getTestStream(batch_size=1).get_epoch_iterator()
#
#        while True:
#            # shape (1, image_features), (1, character sequence length)
#            mnist_im, mnist_txt_enc = ep.next()
#            # get the first one
#            #mnist_im = mnist_im[0]
#            mnist_txt_enc = mnist_txt_enc[0]
#            mnist_txt = "".join(code2char[code] for code in mnist_txt_enc)
#            print "Target: ", mnist_txt
#            message = ("Enter the number of samples\n" if mode == "sample"
#                        else "Enter the beam size\n")
#            batch_size = int(input(message))
#            f_gen_out = f_gen(mnist_im)
#            #sample = ComputationGraph(generator.generate(mnist_im)).get_theano_function()
#            #states, outputs, costs = [data[:, 0] for data in sample()]
#
#            import ipdb; ipdb.set_trace();
#            samples, costs = generate(
#                    np.repeat(np.array(mnist_im),
#                          batch_size, axis=0)
#                    )
#            messages = []
#            for sample, cost in equizip(samples, costs):
#                message = "({})".format(cost)
#                message += "".join(code2char[code] for code in sample)
#                messages.append((cost, message))
#            messages.sort(key=operator.itemgetter(0), reverse=True)
#            for _, message in messages:
#                print(message)

# # # # # # #
#  Model IO #
# # # # # # #


class ModelIO():

    @staticmethod
    def save(func, saveto):
        pkl.dump(func, open('%s.pkl'%saveto, 'wb'))

    @staticmethod
    def load(saveto):
        func = pkl.load(open('%s.pkl'%saveto, 'r'))
        return func

if __name__ == "__main__":
    main("train", "predict-mnist/pklmain_loop")
    main("sample", "predict-mnist/pklmain_loop")

