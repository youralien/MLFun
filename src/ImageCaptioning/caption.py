# scientific python
import theano
from theano import tensor as T

# blocks model building
from blocks.initialization import IsotropicGaussian, Constant
from blocks.graph import ComputationGraph

# blocks model training
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent, AdaDelta
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions import Printing, ProgressBar, FinishAfter

from fuel.transformers import Merge
from fuel.schemes import SequentialScheme

# foxhound
from foxhound.preprocessing import Tokenizer
from foxhound import iterators
from foxhound.theano_utils import floatX, intX
from foxhound.transforms import SeqPadded

# local imports
from modelbuilding import Encoder, PairwiseRanking
from dataset import (coco, FoxyDataStream, GloveTransformer,
    ShuffleBatch, FoxyIterationScheme)

# # # # # # # # # # #
# DataPreprocessing #
# # # # # # # # # # #

trX, teX, trY, teY = coco(mode="dev")

# Word Vectors
vect = Tokenizer(min_df=1, max_features=1000)
vect.fit(trY)

# theano.config.compute_test_value = 'off' # Use 'warn' to activate this feature

# Transforms
trXt=lambda x: floatX(x)
teXt=lambda x: floatX(x)
Yt=lambda y: intX(SeqPadded(vect.transform(y), 'back'))

# Foxhound Iterators
train_batch_size = 16
test_batch_size = 16

train_iterator = [iterators.Linear(
    trXt=trXt, trYt=Yt, size=train_batch_size, shuffle=False
    ) for i in range(2)
]
train_iterator_k = [iterators.Linear(
    trXt=trXt, trYt=Yt, size=train_batch_size, shuffle=False
    ) for i in range(2)
]

# DataStreams
sources = ("image_vects", "word_vects")
sources_k = ("image_vects_k", "word_vects_k")

train_stream = [FoxyDataStream(
      (trX, trY)
    , sources
    , train_iterator[i]
    , FoxyIterationScheme(len(trX), train_batch_size)
    ) for i in range(2)
]
train_stream_k = [FoxyDataStream(
      (trX, trY)
    , sources_k
    , train_iterator_k[i]
    , FoxyIterationScheme(len(trX), train_batch_size)
    ) for i in range(2)
]

# Glove Word Vectors
embedding_dim = 50 # embedding dimension = K.  50 or 300
glove_version = "glove.6B.%sd.txt.gz" % embedding_dim

train_transformer = [GloveTransformer(
    glove_version, data_stream=train_stream[i], vectorizer=vect
    ) for i in range(2)
]
train_transformer_k = [GloveTransformer(
    glove_version, data_stream=train_stream_k[i], vectorizer=vect
    ) for i in range(2)
]

# Final Data Streams w/ contrastive examples
final_train_stream = [Merge(
      (train_transformer[i], ShuffleBatch(train_transformer_k[i]))
    , sources + sources_k
    ) for i in range(2)
]
for stream in final_train_stream:
    stream.iteration_scheme = FoxyIterationScheme(len(trX), train_batch_size)

# # # # # # # # # # #
# Modeling Building #
# # # # # # # # # # #

s = Encoder(
          image_feature_dim=4096
        , embedding_dim=embedding_dim
        , biases_init=Constant(0.)
        , weights_init=IsotropicGaussian(0.02)
        )
s.initialize()

image_vects = T.matrix('image_vects') # named to match the source name
word_vects = T.tensor3('word_vects') # named to match the source name
image_vects_k = T.matrix('image_vects_k') # named to match the contrastive source name
word_vects_k = T.tensor3('word_vects_k') # named to match the contrastive source name

# import numpy as np
# image_vects.tag.test_value = np.zeros((2, 4096), dtype='float32')
# word_vects.tag.test_value = np.zeros((2, 15, 50), dtype='float32')
# image_vects_k.tag.test_value = np.zeros((2, 4096), dtype='float32')
# word_vects_k.tag.test_value = np.zeros((2, 15, 50), dtype='float32')

# x is image_embedding matrix, v is the hidden states representing the sentences
X, V = s.apply(image_vects, word_vects)
X_k, V_k = s.apply(image_vects_k, word_vects_k)

cost = PairwiseRanking(alpha=0.2).apply(X, V, X_k, V_k)

cg = ComputationGraph(cost)

# # # # # # # # # # #
# Modeling Training #
# # # # # # # # # # #

import ipdb
ipdb.set_trace()

algorithm = GradientDescent(
      cost=cost
    , parameters=cg.parameters
    , step_rule=AdaDelta()
    )
main_loop = MainLoop(
      model=Model(cost)
    , data_stream=final_train_stream[0]
    , algorithm=algorithm
    , extensions=[
          DataStreamMonitoring(
              [cost]
            , final_train_stream[1]
            , prefix='train')
       # , DataStreamMonitoring(
       #       [cost]
       #     , final_test_k_stream
       #     , prefix='test')
        , Printing()
        , ProgressBar()
        ]
    )
main_loop.run()
