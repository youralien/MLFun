# scientific python
from theano import tensor as T

# blocks model building
from blocks.initialization import IsotropicGaussian, Constant
from blocks.graph import ComputationGraph

# blocks model training
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent, AdaDelta
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions import Printing, ProgressBar

from fuel.transformers import Merge

# foxhound
from foxhound.preprocessing import Tokenizer
from foxhound import iterators
from foxhound.theano_utils import floatX, intX
from foxhound.transforms import SeqPadded

# local imports
from modelbuilding import Encoder, PairwiseRanking
from dataset import coco, FoxyDataStream, GloveTransformer



# # # # # # # # # # # 
# Modeling Training #
# # # # # # # # # # #
trX, teX, trY, teY = coco('dev')

# Word Vectors
vect = Tokenizer(min_df=1, max_features=1000)
vect.fit(trY)

# Transforms
trXt=lambda x: floatX(x)
teXt=lambda x: floatX(x)
Yt=lambda y: intX(SeqPadded(vect.transform(y), 'back'))
# Yt=lambda y: y


# Foxhound Iterators
train_iterator = iterators.Linear(trXt=trXt, trYt=Yt, shuffle=True)
test_iterator = iterators.Linear(trXt=teXt, trYt=Yt, shuffle=True)

# the batch will be shuffled differently than the train_iterator, which is ideal.
contrastive_iterator = iterators.Linear(trXt=trXt, trYt=Yt, shuffle=True)


# DataStreams
train_stream = FoxyDataStream((trX, trY), ("image_vects", "word_vects"), train_iterator)
test_stream = FoxyDataStream((teX, teY), ("image_vects", "word_vects"), test_iterator)
# contrastive_stream = FoxyDataStream(trX, trY, "image_vects_k", "word_vects_k", contrastive_iterator)
# stream = Merge((train_stream, test_stream), ("image_vects", "word_vects", "image_vects_k", "word_vects_k"))
# image_vects, tokens = train_stream.get_epoch_iterator().next()

# print trY
# print "\n"
# print vect.inverse_transform(tokens.T)

# embedding_dim = K.  50 or 300
embedding_dim = 50

glove_version = "glove.6B.%sd.txt.gz" % embedding_dim
train_transformer = GloveTransformer(glove_version, data_stream=train_stream, vectorizer=vect)
test_transformer = GloveTransformer(glove_version, data_stream=test_stream, vectorizer=vect)
# contrastive_transformer = GloveTransformer(glove_version, data_stream=contrastive_stream, vectorizer=vect)
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
# image_vects_k = T.matrix('image_vects_k') # named to match the contrastive source name
# word_vects_k = T.tensor3('word_vects_k') # named to match the contrastive source name

# x is image_embedding matrix, v is the hidden states representing the sentences
X, V = s.apply(image_vects, word_vects)
# X_k, V_k = s.apply(image_vects_k, word_vects_k)

cost = PairwiseRanking(alpha=0.2).apply(X, V, X, V)

cg = ComputationGraph(cost)

# Train
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
            , train_transformer,
              prefix='train')
        # , DataStreamMonitoring(
        #       [cost]
        #     , test_stream,
        #       prefix='test')
        , Printing()
        , ProgressBar()
        ]
    )
main_loop.run()
