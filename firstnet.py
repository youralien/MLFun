import numpy as np
from theano import tensor

# blocks model building
from blocks.bricks import MLP, Rectifier, Softmax
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.cost import MisclassificationRate
from blocks.graph import ComputationGraph
from blocks.bricks.conv import ConvolutionalLayer, ConvolutionalSequence
from blocks.bricks.conv import Flattener
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT, FILTER


# blocks model training
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions import FinishAfter, Printing, ProgressBar

# fuel
from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

def main():
    # # # # # # # # # # # 
    # Modeling Building #
    # # # # # # # # # # # 
    
    # ConvOp requires input be a 4D tensor
    x = tensor.tensor4("features")

    y = tensor.lmatrix("targets")

    # Convolutional Layers
    # ====================
    conv_layers = [
          ConvolutionalLayer(Rectifier().apply, (5,5), 16, (2,2), name='l1')
        , ConvolutionalLayer(Rectifier().apply, (5,5), 32, (2,2), name='l2')
        ]
    
    convnet = ConvolutionalSequence(
        conv_layers, num_channels=1, image_size=(28,28),
        weights_init=IsotropicGaussian(0.1),
        biases_init=Constant(0)
        )
    convnet.initialize()

    output_dim = np.prod(convnet.get_dim('output'))

    # Fully Connected Layers
    # ======================

    features = Flattener().apply(convnet.apply(x))

    mlp = MLP(  activations=[Rectifier(), None]
              , dims=[output_dim, 100, 10]
              , weights_init=IsotropicGaussian(0.01)
              , biases_init=Constant(0)
        )
    mlp.initialize()

    y_hat = mlp.apply(features)

    # Numerically Stable Softmax
    cost = Softmax().categorical_cross_entropy(y.flatten(), y_hat)
    error_rate = MisclassificationRate().apply(y.flatten(), y_hat)

    cg = ComputationGraph(cost)

    weights = VariableFilter(roles=[FILTER, WEIGHT])(cg.variables)
    l2_regularization = 0.005 * sum((W**2).sum() for W in weights)

    cost = cost + l2_regularization
    cost.name = 'cost_with_regularization'

    # Print sizes to check
    print("Representation sizes:")
    for layer in convnet.layers:
        print(layer.get_dim('input_'))

    # # # # # # # # # # # 
    # Modeling Training #
    # # # # # # # # # # # 

    # Figure out data source
    train = MNIST("train")
    test = MNIST("test")

    # Load Data Using Fuel
    train_stream = DataStream.default_stream(
          dataset=train
        , iteration_scheme=SequentialScheme(train.num_examples, batch_size=128))
    test_stream = DataStream.default_stream(
          dataset=test
        , iteration_scheme=SequentialScheme(test.num_examples, batch_size=1024))

    # Train
    algorithm = GradientDescent(
          cost=cost
        , params=cg.parameters
        , step_rule=Scale(learning_rate=0.1)
        )

    main_loop = MainLoop(
          model=Model(cost)
        , data_stream=train_stream
        , algorithm=algorithm
        , extensions=[
              FinishAfter(after_n_epochs=5)
            , DataStreamMonitoring(
                  [cost, error_rate]
                , test_stream,
                  prefix='test')
            , Printing()
            , ProgressBar()
            ]
        )
    main_loop.run()

if __name__ == '__main__':
    main()

