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
from dataset import coco, FoxyDataStream, GloveTransformer, ShuffleBatch



# # # # # # # # # # #
# Modeling Training #
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
# Yt=lambda y: y


# Foxhound Iterators
train_batch_size = 16
test_batch_size = 16

train_iterator = iterators.Linear(trXt=trXt, trYt=Yt, size=train_batch_size, shuffle=False)

# train_iterator = lambda x: iterators.Linear(trXt=trXt, trYt=Yt, size=train_batch_size, shuffle=False)
train_iterator2 = iterators.Linear(trXt=trXt, trYt=Yt, size=train_batch_size, shuffle=False)
# train_iterator_k = lambda x: iterators.Linear(trXt=trXt, trYt=Yt, size=train_batch_size, shuffle=False)
# test_iterator = lambda x: iterators.Linear(trXt=teXt, trYt=Yt, size=test_batch_size, shuffle=False)
# test_iterator_k = lambda x: iterators.Linear(trXt=teXt, trYt=Yt, size=test_batch_size, shuffle=False)

# DataStreams
train_stream = FoxyDataStream(
      (trX, trY)
    , ("image_vects", "word_vects")
    , train_iterator
    )
train_stream2 = FoxyDataStream(
      (trX, trY)
    , ("image_vects", "word_vects")
    , train_iterator2
    )
# test_stream = FoxyDataStream(
#       (teX, teY)
#     , ("image_vects", "word_vects")
#     , test_iterator
#     )
#test_k_stream = FoxyDataStream(
#      (teX, teY)
#    , ("image_vects", "word_vects")
#    , test_k_iterator
#    , SequentialScheme(len(teX), test_batch_size)
#    )

# image_vects, tokens = train_stream.get_epoch_iterator().next()
# print trY[:3]
# print "\n"
# print vect.inverse_transform(tokens.T[:3])


# embedding_dim = K.  50 or 300
embedding_dim = 50

glove_version = "glove.6B.%sd.txt.gz" % embedding_dim
train_transformer = GloveTransformer(glove_version, data_stream=train_stream, vectorizer=vect)
train_transformer2 = GloveTransformer(glove_version, data_stream=train_stream2, vectorizer=vect)
# test_transformer = GloveTransformer(glove_version, data_stream=test_stream, vectorizer=vect)
#train_k_transformer = GloveTransformer(glove_version, data_stream=train_k_stream, vectorizer=vect)
#test_k_transformer = GloveTransformer(glove_version, data_stream=test_k_stream, vectorizer=vect)
# ep = train_transformer.get_epoch_iterator()


"""
image_vects: array-like, shape (n_examples, 4096)
word_vects: lists of list of lists, shape-ish (n_examples, n_words, embedding dimensionality (50 or 300))
image_vects, word_vects = transformer.get_epoch_iterator().next()
"""

s = Encoder(
          image_feature_dim=4096
        , embedding_dim=embedding_dim
        , biases_init=Constant(0.)
        , weights_init=IsotropicGaussian(0.02)
        )
s.initialize()

image_vects = T.matrix('image_vects') # named to match the source name
word_vects = T.tensor3('word_vects') # named to match the source name
#image_vects_k = T.matrix('image_vects_k') # named to match the contrastive source name
#word_vects_k = T.tensor3('word_vects_k') # named to match the contrastive source name

# import numpy as np
# image_vects.tag.test_value = np.zeros((2, 4096), dtype='float32')
# word_vects.tag.test_value = np.zeros((2, 15, 50), dtype='float32')
# image_vects_k.tag.test_value = np.zeros((2, 4096), dtype='float32')
# word_vects_k.tag.test_value = np.zeros((2, 15, 50), dtype='float32')

# x is image_embedding matrix, v is the hidden states representing the sentences
X, V = s.apply(image_vects, word_vects)
#X_k, V_k = s.apply(image_vects_k, word_vects_k)

cost = PairwiseRanking(alpha=0.2).apply(X, V, X, V)

cg = ComputationGraph(cost)

#final_train_k_stream = Merge(
#      (train_transformer, ShuffleBatch(train_k_transformer))
#    , ("image_vects", "word_vects", "image_vects_k", "word_vects_k")
#    )
#
#final_test_k_stream = Merge(
#      (test_transformer, ShuffleBatch(test_k_transformer))
#    , ("image_vects", "word_vects", "image_vects_k", "word_vects_k")
#    )

#import ipdb
#trep = final_train_k_stream.get_epoch_iterator()
#train = trep.next()
#teep = final_train_k_stream.get_epoch_iterator()
#test = teep.next()
#print [t.shape for t in train]
#print [t.shape for t in test]
#
#ipdb.set_trace()

import ipdb
ipdb.set_trace()

algorithm = GradientDescent(
      cost=cost
    , parameters=cg.parameters
    , step_rule=AdaDelta()
    )
main_loop = MainLoop(
      model=Model(cost)
    , data_stream=train_transformer
    , algorithm=algorithm
    , extensions=[
          DataStreamMonitoring(
              [cost]
            , train_transformer2
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
