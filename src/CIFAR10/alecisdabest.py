import numpy as np

from fuel.datasets import CIFAR10
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

from foxhound.models import Network
from foxhound import ops
from foxhound import iterators
from foxhound.transforms import OneHot Fliplr, Patch, CenterCrop
from foxhound.theano_utils import floatX

from utils import misclassification_rate

# # FCC
# model = [
#     ops.Input(['x', 28 * 28])
#   , ops.Project(dim=512)
#   , ops.Activation('rectify')
#   , ops.Project(dim=512)
#   , ops.Activation('rectify')
#   , ops.Project(dim=10)
#   , ops.Activation('softmax')
#   ]

# Network to Network
model = [
      ops.Input(['x', 3, 28, 28])
    , ops.Conv(64, (3,3))
    , ops.Activation('rectify')
    , ops.Conv(64, (3,3))
    , ops.Activation('rectify')
    , ops.CUDNNPool((3,3), (2,2), pad=(1,1))
    , ops.Dropout(.10)
    , ops.Conv(128, (3,3))
    , ops.Activation('rectify')
    , ops.Conv(128, (3,3))
    , ops.Activation('rectify')
    , ops.CUDNNPool((3,3), (2,2), pad=(1,1))
    , ops.Dropout(.15)
    , ops.Conv(256, (3,3))
    , ops.Activation('rectify')
    , ops.Conv(256, (3,3))
    , ops.Activation('rectify')
    , ops.Conv(256, (3,3))
    , ops.Activation('rectify')
    , ops.Conv(256, (3,3))
    , ops.Activation('rectify')
    , ops.CUDNNPool((3,3), (2,2), pad=(1,1))
    , ops.Dropout(.25)
    , ops.Flatten(2)
    , ops.Project(dim=4096)
    , ops.Activation('rectify')
    , ops.Dropout(0.5)
    , ops.Project(dim=10)
    , ops.Activation('softmax')
    ]


# # # # # # # # # # # 
# Modeling Training #
# # # # # # # # # # # 

# Figure out data source
train = CIFAR10("train")
test = CIFAR10("test")

# Load Data Using Fuel
train_stream = DataStream.default_stream(
      dataset=train
    , iteration_scheme=ShuffledScheme(train.num_examples, batch_size=128))
test_stream = DataStream.default_stream(
      dataset=test
    , iteration_scheme=ShuffledScheme(test.num_examples, batch_size=1024))

train_epoch, test_epoch = [stream.get_epoch_iterator() for stream in [train_stream, test_stream]]


trXt = lambda x: floatX(Fliplr(Patch(np.asarray(x).transpose(0, 2, 3, 1), 28, 28))).transpose(0, 3, 1, 2)
teXt = lambda x: floatX(CenterCrop(np.asarray(x).transpose(0, 2, 3, 1), 28, 28)).transpose(0, 3, 1, 2)
trYt = lambda y: floatX(OneHot(y, 10))
iterator = iterators.Linear(trXt=trXt, teXt=teXt, trYt=trYt)

def get_entire_stream(epoch_iterator):
    Xs = []
    Ys = []
    for xmb, ymb in epoch_iterator:
        Xs.append(xmb)
        Ys.append(ymb)
    X = np.vstack(Xs)
    Y = np.hstack(Ys)
    return X, Y


trX, trY = get_entire_stream(train_epoch)
teX, teY = get_entire_stream(test_epoch) 

# Learn and Predict
model = Network(model, iterator=iterator)

# Keep Running For Infinite Iterations Until a Keyboard Interrupt
continue_epochs = True
min_cost_delta = .00001
min_cost = .001
cost0, cost1 = None, None
epoch_count = 0

while continue_epochs:
    epoch_count += 1
    costs = model.fit(trX, trY)
    if cost0 is None:
        cost0 = costs[-1]
    elif cost1 is None:
        cost1 = costs[-1]
    else:
        if ( (cost1 - cost0) <= min_cost_delta ) and (cost1 <= min_cost):
            continue_epochs = False
    # Eval Train/Test Error Every N Epochs
    if epoch_count % 10 == 0:
        trYpred = np.argmax(model.predict(trX), axis=1)
        teYpred = np.argmax(model.predict(teX), axis=1)
        train_error = misclassification_rate(trY, trYpred)
        test_error = misclassification_rate(teY, teYpred)
        print "Train Error: ", train_error
        print "Test Error: ", test_error

