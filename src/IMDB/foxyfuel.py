
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

	Y: array-like, shape (n_samples, n_classes)
		targets

	iterator: a Foxhound iterator.  The use is jank right now, but always use
		trXt and trYt as the X and Y transforms respectively
	"""

	def __init__(self, X, Y, iterator, iteration_scheme=None):
		self.X = X
		self.Y = Y
		self.iterator = iterator
		self.iteration_scheme = iteration_scheme # Compatibility with the blocks mainloop

	def get_epoch_iterator(self, as_dict=False):

		for xmb, ymb in self.iterator.iterXY(self.X, self.Y):
			yield {"X": xmb, "Y": ymb} if as_dict else (xmb, ymb)