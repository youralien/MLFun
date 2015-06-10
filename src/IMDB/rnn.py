from imdb import imdb

from foxhound.models import Network
from foxhound.preprocessing import Tokenizer
from foxhound import ops
from foxhound import iterators
from foxhound.theano_utils import floatX, intX
from foxhound.transforms import SeqPadded, LenClip

from utils import misclassification_rate
import ipdb

trX, teX, trY, teY = imdb()

# min_df==min document frequency
vect = Tokenizer(min_df=10, max_features=100000)
vect.fit(trX)

ops = [
	# time dimension, 
	  ops.Input([100, 'x'], dtype='int32')
	, ops.Embedding(256, n_embed=vect.n_features)
	, ops.GRU(256)
	, ops.Slice()
	, ops.Project(128)
	, ops.Activation('rectify')
	, ops.Project(1)
	, ops.Activation('sigmoid')
]

# 
iterator = iterators.Linear(
	  trXt=lambda x: intX(SeqPadded(vect.transform(LenClip(x, 1000))))
	, teXt=lambda x: intX(SeqPadded(vect.transform(x)))
	, trYt=lambda y: floatX(y).reshape(-1, 1)
	)

# Learn and Predict
model = Network(ops, iterator=iterator)

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
        trYpred = model.predict(trX)
        teYpred = model.predict(teX)
        trY = trY > .5
        teY = teY > .5
        trYpred = trYpred > .5
        teYpred = teYpred > .5
        train_error = misclassification_rate(trY, trYpred)
        test_error = misclassification_rate(teY, teYpred)
        print "Train Error: ", train_error
        print "Test Error: ", test_error
