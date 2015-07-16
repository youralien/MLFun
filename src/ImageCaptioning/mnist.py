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
        try:
            ModelIO.save(f_gen, 'predict-mnist/f_gen')
            print "It actually saved.  Thank you pickle!"
        except Exception, e:
            print "Fuck Pickle and continue :)"
            print e
        ModelEval.predict(f_gen)
    else:
        logger.info("Loading Model ...")
        f_gen = ModelIO.load('predict-mnist/f_gen')
        ModelEval.predict(f_gen)

class ModelEval():

    @staticmethod
    def predict(f_gen):
        ep = getTestStream(batch_size=1).get_epoch_iterator()
        while True:
            im_vects, txt_enc = ep.next()
            mnist_txt = "".join(code2char[code] for code in txt_enc[0])
            print "\nTrying for: ", mnist_txt
            message=("Number of attempts to generate correct text? ")
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
    #main("train", "predict-mnist/pklmain_loop")
    main("sample", "predict-mnist/pklmain_loop")

