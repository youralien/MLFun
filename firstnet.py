# pure theano
from theano import tensor

# blocks model building
from blocks.bricks import MLP, Rectifier, Softmax
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.graph import ComputationGraph

# blocks model training
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions import FinishAfter, Printing

# fuel
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

def main():

	# Construct Model
	FULLY_CONNECTED_INPUT_SIZE = 784
	mlp = MLP(  activations=[Rectifier(), Softmax()]
			  , dims=[FULLY_CONNECTED_INPUT_SIZE, 100, 10]
			  , weights_init=IsotropicGaussian(0.01)
			  , biases_init=Constant(0)
		)
	mlp.initialize()

	# Calculate Loss
	x = tensor.matrix("features")
	y = tensor.lmatrix("targets")
	y_hat = mlp.apply(x)
	cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
	error_rate = MisclassificationRate().apply(y.flatten(), y_hat)

	# Figure out data source
	train = MNIST("train")
	test = MNIST("test")

	# Load Data Using Fuel
	train_stream = Flatten(DataStream.default_stream(
		  dataset=train
		, iteration_scheme=SequentialScheme(train.num_examples, batch_size=128)))
	test_stream = Flatten(DataStream.default_stream(
		  dataset=test
		, iteration_scheme=SequentialScheme(test.num_examples, batch_size=1024)))

	# Train
	monitor = DataStreamMonitoring(
		  variables=[cost, error_rate]
		, data_stream=test_stream
		, prefix="test")

	main_loop = MainLoop(
		  model=Model(cost)
		, data_stream=train_stream
		, algorithm=GradientDescent(
			  cost=cost
			, params=ComputationGraph(cost).parameters
			, step_rule=Scale(learning_rate=0.1))
		, extensions=[  FinishAfter(after_n_epochs=5)
					  , monitor
					  , Printing()])
	main_loop.run()

if __name__ == '__main__':
	main()

