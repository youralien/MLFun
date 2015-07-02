from imdb import imdb

# blocks model building
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.bricks.cost import MisclassificationRate
from blocks.graph import ComputationGraph

# blocks model training
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent, AdaDelta
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions import Printing, ProgressBar

from foxhound import iterators
from foxhound.preprocessing import Tokenizer
from foxhound.theano_utils import floatX, intX
from foxhound.transforms import SeqPadded, LenClip

from foxyfuel import FoxyDataStream

from modelbuilding import Sentiment
from theano import tensor as T

# Data Loading
trX, teX, trY, teY = imdb()

# # Word Vectors
vect = Tokenizer(min_df=10, max_features=1000)
vect.fit(trX)

s = Sentiment(
		  dimension=256
		, dictionary_size=vect.n_features
		, biases_init=Constant(0.)
		, weights_init=IsotropicGaussian(0.02) 
		)
s.initialize()

x = T.imatrix('X')
y = T.matrix('Y')
y_hat = s.apply(x)

cost = BinaryCrossEntropy().apply(y, y_hat)
"""Missclassification rate will calculate the mistake as

mistakes = tensor.neq(y, y_hat.argmax(axis=1))

which assumes that y_hat is shape (n_examples, n_classes).
This doesn't always work when our y is not one-hotted

# error_rate = MisclassificationRate().apply(y, y_hat)
"""
cg = ComputationGraph(cost)

# # # # # # # # # # # 
# Modeling Training #
# # # # # # # # # # #

# Transforms
trXt=lambda x: intX(SeqPadded(vect.transform(LenClip(x, 100))))
teXt=lambda x: intX(SeqPadded(vect.transform(x)))
Yt=lambda y: floatX(y).reshape(-1, 1)

# Foxhound Iterators
train_iterator = iterators.Linear(trXt=trXt, trYt=Yt)
test_iterator = iterators.Linear(trXt=teXt, trYt=Yt)

# DataStreams
train_stream = FoxyDataStream(trX, trY, train_iterator)
test_stream = FoxyDataStream(teX, teY, test_iterator)

# import ipdb
# ipdb.set_trace()
# Train
algorithm = GradientDescent(
      cost=cost
    , params=cg.parameters
    , step_rule=AdaDelta()
    )
main_loop = MainLoop(
      model=Model(cost)
    , data_stream=train_stream
    , algorithm=algorithm
    , extensions=[
          DataStreamMonitoring(
              [cost]
            , train_stream,
              prefix='train')
        # , DataStreamMonitoring(
        #       [cost, error_rate]
        #     , test_stream,
        #       prefix='test')
        , Printing()
        , ProgressBar()
        ]
    )
main_loop.run()
