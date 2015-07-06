# # # # # # # # # # # 
# Modeling Building #
# # # # # # # # # # # 
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import GatedRecurrent, LSTM
from blocks.bricks import Initializable, Linear, Logistic

class Sentiment(Initializable):

	def __init__(self, dimension, dictionary_size, rnn_unit='lstm', **kwargs):
		super(Sentiment, self).__init__(**kwargs)

		self.rnn_unit = rnn_unit
		self.embedding = LookupTable(dictionary_size, dimension)

		if self.rnn_unit == 'gru':

			self.to_gate_inputs = Linear(
				  input_dim=dimension
				, output_dim=dimension*2 # gate_inputs = vstack(updates, resets) where updates and resets are shape dimension
				, weights_init=IsotropicGaussian(0.02)
				, biases_init=Constant(0.)
				)

			# Don't think this dim has to also be dimension, more arbitrary
			self.transition = GatedRecurrent(
				dim=dimension, name="transition")

			self.to_score = Linear(input_dim=256, output_dim=1, name="to_score")
			self.to_probs = Logistic()
			
			self.children = [ self.embedding
							, self.to_gate_inputs
							, self.transition
							, self.to_score
							, self.to_probs]
		else:

			self.to_inputs = Linear(
				  input_dim=dimension
				, output_dim=dimension*4 # gate_inputs = vstack(input, forget, cell, hidden)
				, weights_init=IsotropicGaussian(0.02)
				, biases_init=Constant(0.)
				)

			# Don't think this dim has to also be dimension, more arbitrary
			self.transition = LSTM(
				dim=dimension, name="transition")

			self.to_score = Linear(input_dim=256, output_dim=1, name="to_score")
			self.to_probs = Logistic()
			self.children = [ self.embedding
							, self.to_inputs
							, self.transition
							, self.to_score
							, self.to_probs]

	# @application(inputs=['x'], outputs=['gate'])
	@application(inputs=['x'], outputs=['probs'])	
	def apply(self, x):
		embedding = self.embedding.apply(x)
		
		
		if self.rnn_unit == 'gru':

			gate_inputs = self.to_gate_inputs.apply(embedding)
			hidden = self.transition.apply(inputs=embedding, gate_inputs=gate_inputs, mask=None)		

		else:
			inputs = self.to_inputs.apply(embedding)
			hidden, cells = self.transition.apply(inputs=inputs, mask=None)
		
		to_score = self.to_score.apply(hidden[-1])
		return self.to_probs.apply(to_score)

if __name__ == '__main__':
	import theano.tensor as T
	s = Sentiment(
			  dimension=256
			, dictionary_size=1000
			, biases_init=Constant(0.)
			, weights_init=IsotropicGaussian(0.02)
			)
	s.initialize()

	x = T.imatrix('x')
	y = T.vector('y')
	y_hat = s.apply(x)
