# scientific python
import theano
from theano import tensor as T

# blocks model building
from blocks.initialization import Uniform, IsotropicGaussian, Constant
from blocks.graph import ComputationGraph

# blocks model training
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent, AdaDelta
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions import Printing, ProgressBar, FinishAfter

from fuel.transformers import Merge

# foxhound
from foxhound.preprocessing import Tokenizer
from foxhound import iterators
from foxhound.theano_utils import floatX, intX
from foxhound.transforms import SeqPadded

# local imports
from modelbuilding import Encoder, l2norm
from dataset import (coco, FoxyDataStream, GloveTransformer,
    ShuffleBatch, FoxyIterationScheme)

# # # # # # # # # # #
# DataPreprocessing #
# # # # # # # # # # #
batch_size = 128

trX, teX, trY, teY = coco(mode="dev", batch_size=batch_size)
trX_k, teX_k, trY_k, teY_k = (trX, teX, trY, teY)

# trX_k, teX_k, trY_k, teY_k = coco(mode="dev", batch_size=batch_size)

# Word Vectors
vect = Tokenizer(min_df=5, max_features=50000)
vect.fit(trY)

# theano.config.compute_test_value = 'off' # Use 'warn' to activate this feature

# Transforms
trXt=lambda x: floatX(x)
teXt=lambda x: floatX(x)
Yt=lambda y: intX(SeqPadded(vect.transform(y), 'back'))

# Foxhound Iterators


train_iterator = [iterators.Linear(
    trXt=trXt, trYt=Yt, size=batch_size, shuffle=False
    ) for i in range(2)
]
train_iterator_k = [iterators.Linear(
    trXt=trXt, trYt=Yt, size=batch_size, shuffle=False
    ) for i in range(2)
]

# DataStreams
sources = ("image_vects", "word_vects")
sources_k = ("image_vects_k", "word_vects_k")

train_stream = [FoxyDataStream(
      (trX, trY)
    , sources
    , train_iterator[i]
    , FoxyIterationScheme(len(trX), batch_size)
    ) for i in range(2)
]
train_stream_k = [FoxyDataStream(
      (trX_k, trY_k)
    , sources_k
    , train_iterator_k[i]
    , FoxyIterationScheme(len(trX), batch_size)
    ) for i in range(2)
]

# Glove Word Vectors
embedding_dim = 300 # embedding dimension = K.  50 or 300
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
    stream.iteration_scheme = FoxyIterationScheme(len(trX), batch_size)

# # # # # # # # # # #
# Modeling Building #
# # # # # # # # # # #

s = Encoder(
          image_feature_dim=4096
        , embedding_dim=embedding_dim
        , biases_init=Constant(0.)
        , weights_init=Uniform(width=0.08)
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

# learned image embedding, learned sentence embedding
lim, ls = s.apply(image_vects, word_vects)

# learned constrastive im embedding, learned contrastive s embedding
lcim, lcs = s.apply(image_vects_k, word_vects_k)

# l2norms
lim = l2norm(lim)
lcim = l2norm(lcim)
ls = l2norm(ls)
lcs = l2norm(lcs)

# this step unexpected.?
# tile by number of contrastive terms
# lim = T.tile(lim, (len(trX_k), 1))
# ls = T.tile(ls, (len(trX_k), 1))

margin = 0.2 # alpha term, should not be more than 1!

# pairwise ranking loss (https://github.com/youralien/skip-thoughts/blob/master/eval_rank.py)
cost_im = margin - (lim * ls).sum(axis=1) + (lim * lcs).sum(axis=1)
cost_im = cost_im * (cost_im > 0.) # this is like the max(0, pairwise-ranking-loss)
cost_im = cost_im.sum(0)

cost_s = margin - (ls * lim).sum(axis=1) + (ls * lcim).sum(axis=1)
cost_s = cost_s * (cost_s > 0.) # this is like max(0, pairwise-ranking-loss)
cost_s = cost_s.sum(0)

cost = cost_im + cost_s
cost.name = "pairwise ranking loss"

cg = ComputationGraph(cost)

# # # # # # # # # # #
# Modeling Training #
# # # # # # # # # # #

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
        , FinishAfter(after_n_epochs=10)
        ]
    )
main_loop.run()
