# # # # # # # # # # # 
# Modeling Building #
# # # # # # # # # # # 
from abc import ABCMeta, abstractmethod
from six import add_metaclass

from theano import tensor

from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.base import application, Brick
from blocks.bricks.recurrent import LSTM
from blocks.bricks import Initializable, Linear

class Encoder(Initializable):

    def __init__(self, image_feature_dim, embedding_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.image_embedding = Linear(
              input_dim=image_feature_dim
            , output_dim=embedding_dim
            # , weights_init=IsotropicGaussian(0.02)
            # , biases_init=Constant(0.)
            , name="image_embedding"
            )

        self.to_inputs = Linear(
              input_dim=embedding_dim
            , output_dim=embedding_dim*4 # gate_inputs = vstack(input, forget, cell, hidden)
            # , weights_init=IsotropicGaussian(0.02)
            # , biases_init=Constant(0.)
            , name="to_inputs"
            )

        # Don't think this dim has to also be dimension, more arbitrary
        self.transition = LSTM(
            dim=embedding_dim, name="transition")

        self.children = [ self.image_embedding
                        , self.to_inputs
                        , self.transition
                        ]

    @application(inputs=['image_vects', 'word_vects'], outputs=['image_embedding', 'sentence_embedding'])   
    def apply(self, image_vects, word_vects):
        
        image_embedding = self.image_embedding.apply(image_vects)
        
        inputs = self.to_inputs.apply(word_vects)
        hidden, cells = self.transition.apply(inputs=inputs, mask=None)

        # the last hidden state represents the accumulation of all the words (i.e. the sentence)
        # grab all batches, grab the last value representing accumulation of the sequence, grab all features
        sentence_embedding = hidden[:,-1, :]

        return image_embedding, sentence_embedding

@add_metaclass(ABCMeta)
class PairwiseCost(Brick):
    @abstractmethod
    @application
    def apply(self, x, v, x_k, v_k):
        pass


@add_metaclass(ABCMeta)
class PairwiseCostMatrix(PairwiseCost):
    """Base class for pairwise costs which can be calculated element-wise.
    Assumes that the data has format (batch, features).
    """
    @application(outputs=["cost"])
    def apply(self, x, v, x_k, v_k):
        return self.cost_matrix(x, v, x_k, v_k).sum(axis=1).mean()

    @abstractmethod
    @application
    def cost_matrix(self, x, v, x_k, v_k):
        pass

class PairwiseRanking(PairwiseCostMatrix):
    
    def __init__(self, alpha, **kwargs):
        super(PairwiseRanking, self).__init__(**kwargs)
        self.alpha = alpha

    @application
    def cost_matrix(self, x, v, x_k, v_k):
        x_cost = tensor.maximum(
            self.alpha - cos_sim(x, v) + cos_sim(x, v_k)
            , 0
            )
        v_cost = tensor.maximum(
            self.alpha - cos_sim(v, x) + cos_sim(v, x_k)
            , 0
            )

        # cannot be broadcasted together if not transposed
        cost = x_cost + v_cost.T

        return cost

def cos_sim(x, v):
    scaled_x = x / tensor.nlinalg.norm(x, ord=1)
    scaled_v = v / tensor.nlinalg.norm(x, ord=1)

    # cannot be dotted together if not transposed
    cosine_similarity = tensor.dot(scaled_x, scaled_v.T)

    return cosine_similarity

if __name__ == '__main__':
    import theano
    import theano.tensor as T
    theano.config.compute_test_value = 'warn'

    image_vects = T.matrix('image_vects')
    word_vects = T.tensor3('word_vects')
    
    import numpy as np

    batch_size = 2
    image_feature_dim = 64
    seq_len = 4
    embedding_dim = 7
    image_vects.tag.test_value = np.zeros((batch_size, image_feature_dim), dtype='float32')
    word_vects.tag.test_value = np.zeros((batch_size, seq_len, embedding_dim), dtype='float32')

    s = Encoder(
              image_feature_dim=image_feature_dim
            , embedding_dim=embedding_dim
            , biases_init=Constant(0.)
            , weights_init=IsotropicGaussian(0.02)
            )
    s.initialize()
    iem, sem = s.apply(image_vects, word_vects)

    # expecting sentence embedding to be [batch_size, embedding_dim]
    f = theano.function([word_vects], sem)
    print(f(word_vects.tag.test_value))