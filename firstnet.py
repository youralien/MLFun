import numpy as np
from theano import tensor

# blocks model building
from blocks.bricks import MLP, Rectifier, Softmax
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.cost import MisclassificationRate
from blocks.graph import ComputationGraph
from blocks.bricks.conv import ConvolutionalActivation, ConvolutionalLayer, ConvolutionalSequence
from blocks.bricks.conv import Flattener
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT, FILTER

# blocks model training
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent, Adam
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions import Printing, ProgressBar

# fuel
from fuel.datasets import CIFAR10
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from extensions import ExperimentSaver

def main():
    # # # # # # # # # # # 
    # Modeling Building #
    # # # # # # # # # # # 
    
    # ConvOp requires input be a 4D tensor
    x = tensor.tensor4("features")

    y = tensor.ivector("targets")

    # Convolutional Layers
    # ====================
    conv_layers = [
        # ConvolutionalLayer(activiation, filter_size, num_filters, pooling_size, name)
          ConvolutionalLayer(Rectifier().apply, (5,5), 64, (2,2), border_mode='full', name='l1')
        , ConvolutionalLayer(Rectifier().apply, (3,3), 128, (2,2), border_mode='full', name='l2')
        , ConvolutionalActivation(Rectifier().apply, (3,3), 256, name='l3')
        , ConvolutionalLayer(Rectifier().apply, (3,3), 256, (2,2), name='l4')
        # , ConvolutionalLayer(Rectifier().apply, (3,3), 128, (2,2), name='l4')
        ]
    
    convnet = ConvolutionalSequence(
        conv_layers, num_channels=3, image_size=(32,32),
        weights_init=IsotropicGaussian(0.1),
        biases_init=Constant(0)
        )
    convnet.initialize()

    output_dim = np.prod(convnet.get_dim('output'))

    # Fully Connected Layers
    # ======================
    conv_features = convnet.apply(x)
    features = Flattener().apply(conv_features)

    mlp = MLP(  activations=[Rectifier()]*2+[None]
              , dims=[output_dim, 256, 256, 10]
              , weights_init=IsotropicGaussian(0.01)
              , biases_init=Constant(0)
        )
    mlp.initialize()

    y_hat = mlp.apply(features)
    # print y_hat.shape.eval({x: np.zeros((1, 3, 32, 32), dtype=theano.config.floatX)})

    # Numerically Stable Softmax
    cost = Softmax().categorical_cross_entropy(y, y_hat)
    error_rate = MisclassificationRate().apply(y, y_hat)

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
    train = CIFAR10("train")
    test = CIFAR10("test")

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
        , step_rule=Adam(learning_rate=0.0005)
        )

    main_loop = MainLoop(
          model=Model(cost)
        , data_stream=train_stream
        , algorithm=algorithm
        , extensions=[
              TrainingDataMonitoring(
                  [cost, error_rate]
                , prefix='train'
                , after_epoch=True)
            , DataStreamMonitoring(
                  [cost, error_rate]
                , test_stream,
                  prefix='test')
            , ExperimentSaver(dest_directory='dest', src_directory='src')
            , Printing()
            , ProgressBar()
            ]
        )
    main_loop.run()

if __name__ == '__main__':
    main()

