from imdb import imdb
from foxhound import iterators
from foxhound.preprocessing import Tokenizer
from foxhound.theano_utils import floatX, intX
from foxhound.transforms import SeqPadded, LenClip


# Foxhound + Fuel
class FoxyDataStream(object):
	"""FoxyDataStream attempts to merge the gap between fuel DataStreams and
	Foxhound iterators.

	The place we will be doing this merge is in the blocks MainLoop. Inserting
	a FoxyDataStream() in place of a DataStream.default_stream()
	will suffice.

	(Note)
	These are broken down into the following common areas
	- dataset which has (features, targets) or (X, Y)
	- iteration_scheme (sequential vs shuffling, batch_size)
	- transforms

	Parameters
	----------
	X: array-like, shape (n_samples, ... )
		features

	Y: array-like, shape (n_samples, )
		targets

	iterator: a Foxhound iterator.  The use is jank right now, but always use
		trXt and trYt as the X and Y transforms respectively
	"""

	def __init__(self, X, Y, iterator):

		self.X = X
		self.Y = Y
		self.iterator = iterator

	def get_epoch_iterator(self, as_dict=False):

		for xmb, ymb in self.iterator.iterXY(self.X, self.Y):
			yield {"X": xmb, "Y": ymb} if as_dict else xmb, ymb

trX, teX, trY, teY = imdb()

vect = Tokenizer(min_df=10, max_features=100000)
vect.fit(trX)

trXt=lambda x: intX(SeqPadded(vect.transform(LenClip(x, 1000))))
teXt=lambda x: intX(SeqPadded(vect.transform(x)))
Yt=lambda y: floatX(y).reshape(-1, 1)

train_iterator = iterators.Linear(trXt=trXt, trYt=Yt)
test_iterator = iterators.Linear(trXt=teXt, trYt=Yt)

train_stream = FoxyDataStream(trX, trY, train_iterator)
test_stream = FoxyDataStream(teX, teY, test_iterator)