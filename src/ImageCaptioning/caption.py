import ipdb

from foxhound.models import Network
from foxhound.preprocessing import Tokenizer
from foxhound import ops
from foxhound import iterators
from foxhound.theano_utils import floatX, intX
from foxhound.transforms import SeqPadded, LenClip

from dataset import coco, FoxyDataStream, GloveTransformer



# # # # # # # # # # # 
# Modeling Training #
# # # # # # # # # # #
trX, teX, trY, teY = coco('dev')

# Transforms
trXt=lambda x: floatX(x)
teXt=lambda x: floatX(x)
Yt=lambda y: y

# Foxhound Iterators
train_iterator = iterators.Linear(trXt=trXt, trYt=Yt)
test_iterator = iterators.Linear(trXt=teXt, trYt=Yt)

# DataStreams
train_stream = FoxyDataStream(trX, trY, train_iterator)
# test_stream = FoxyDataStream(teX, teY, test_iterator)
glove_version = "glove.6B.50d.txt.gz"
transformer = GloveTransformer(glove_version, data_stream=train_stream)
image_vects, word_vects = transformer.get_epoch_iterator().next()
# print train_stream.get_epoch_iterator().next()
# ops = [
# 	# time dimension,
# 	  ops.Input(4096, dtype='float32') # EasyNet Image Features
#     , ops.GRU(512)
# 	, ops.GRU(512)
# 	, ops.Slice() # Kinda like Flatten (Conv -> Fully Connected)==(GRU -> Fully Connected)
# 	, ops.Project(128)
# 	, ops.Activation('rectify')
# 	, ops.Project(Length of Bag Of Words)
# 	, ops.Activation('softmax')
# ]


# iterator = iterators.linear(
#       trXt=lambda x: floatX(x)
#     , teXt=lambda x: floatX(x)
#     , trYt = lambda y: floatX(OneHot(SeqPadded(vect.transform(y), 'back'), Length Of Bag of Words))
#     )


# # iterator = iterators.linear(
# #       trxt=lambda x: intX(seqpadded(vect.transform(x), 'back'))
# #     , text=lambda x: intX(seqpadded(vect.transform(x), 'back'))
# #     , tryt=lambda y: floatx(y).reshape(-1, 1)
# # 	)

# # Learn and Predict
# model = Network(ops, iterator=iterator)

# # Keep Running For Infinite Iterations Until a Keyboard Interrupt
# continue_epochs = True
# min_cost_delta = .00001
# min_cost = .001
# cost0, cost1 = None, None
# epoch_count = 0

# while continue_epochs:
#     ipdb.set_trace()
#     epoch_count += 1
#     costs = model.fit(trX, trY)
#     if cost0 is None:
#         cost0 = costs[-1]
#     elif cost1 is None:
#         cost1 = costs[-1]
#     else:
#         if ( (cost1 - cost0) <= min_cost_delta ) and (cost1 <= min_cost):
#             continue_epochs = False
#     # Eval Train/Test Error Every N Epochs
#     if epoch_count % 50 == 0:
#         trYpred = model.predict(trX)
#         teYpred = model.predict(teX)
        
#         # Generate back into sentences
#         trY = trY > .5
#         teY = teY > .5
#         trYpred = trYpred > .5
#         teYpred = teYpred > .5
#         train_error = misclassification_rate(trY, trYpred)
#         test_error = misclassification_rate(teY, teYpred)
#         print "Train Error: ", train_error
#         print "Test Error: ", test_error
