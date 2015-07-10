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
        
        # inputs = word_vects
        inputs = self.to_inputs.apply(word_vects)
        inputs = inputs.dimshuffle(1, 0, 2)
        hidden, cells = self.transition.apply(inputs=inputs, mask=None)

        # the last hidden state represents the accumulation of all the words (i.e. the sentence)
        # grab all batches, grab the last value representing accumulation of the sequence, grab all features
        sentence_embedding = hidden[-1]
        # sentence_embedding = inputs.mean(axis=0)
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
        cost = x_cost + v_cost

        return cost

# l2 norm, row-wise
def l2norm(X):
    norm = tensor.sqrt(tensor.pow(X, 2).sum(1))
    X /= norm[:, None]
    return X

def cos_sim(x, v):
    """
    Parameters
    ----------
    x shape (batch_size, dim)
    v shape (batch_size, dim)

    Intermediate
    ------------
    cosine_similarity (batch_size, batch_size)
        the cos_sim of batch i of X with batch j of V
    
    Returns
    -------
    cosine_simliarity_matching (batch_size, )
        the cos_sim of batch i of X and i of V
    """

    scaled_x = l2norm(x)
    scaled_v = l2norm(v)
    # scaled_x = x / tensor.nlinalg.norm(x, None)
    # scaled_v = v / tensor.nlinalg.norm(x, None)

    # cannot be dotted together if not transposed
    cosine_similarity = tensor.dot(scaled_x, scaled_v.T)

    cosine_similarity_matching = cosine_similarity.diagonal()
    return cosine_similarity_matching

if __name__ == '__main__':
    import theano
    import theano.tensor as T
    import numpy as np

    def test_encoder():
        image_vects = T.matrix('image_vects')
        word_vects = T.tensor3('word_vects')
        

        batch_size = 2
        image_feature_dim = 64
        seq_len = 4
        embedding_dim = 7

        s = Encoder(
                  image_feature_dim=image_feature_dim
                , embedding_dim=embedding_dim
                , biases_init=Constant(0.)
                , weights_init=IsotropicGaussian(0.02)
                )
        s.initialize()
        iem, sem = s.apply(image_vects, word_vects)

        image_vects_tv = np.zeros((batch_size, image_feature_dim), dtype='float32')
        word_vects_tv = np.zeros((batch_size, seq_len, embedding_dim), dtype='float32')
        # expecting sentence embedding to be [batch_size, embedding_dim]
        f = theano.function([word_vects], sem)
        
        print "input word vects shape: ", (batch_size, seq_len, embedding_dim)
        out = f(word_vects_tv)
        print "output shape"
        print out.shape
        print "output"
        print out

    # test_encoder()
    def test_cos_sim():
        X = T.matrix('iem')
        V = T.matrix('sem')

        s = cos_sim(X, V)

        f = theano.function([X, V], s, allow_input_downcast=True)

        batch_size = 3
        embedding_dim = 7
        shape = (batch_size, embedding_dim)

        print "\n each x, v in X,V are similar, cost should be low"
        X_tv = np.random.rand(*shape)
        V_tv = X_tv.copy()

        print "in: x, v"
        print X_tv
        print V_tv
        print "out: cos sim"
        print f(X_tv, V_tv)

    # test_cos_sim()
    def test_cost():

        margin = 0.2 # alpha term, should not be more than 1!
        batch_size = 5
        embedding_dim = 50
        shape = (batch_size, embedding_dim)

        # image embedding
        lim = T.matrix('lim')
        lcim = T.matrix('lcim')

        # sentence embedding
        ls = T.matrix('ls')
        lcs = T.matrix('lcs')

        # l2norms
        lim = l2norm(lim)
        lcim = l2norm(lcim)
        ls = l2norm(ls)
        lcs = l2norm(lcs)

        # tile by number of contrastive terms
        # lim = tensor.tile(lim, (batch_size, 1))
        # ls = tensor.tile(ls, (batch_size, 1))
        
        # pairwise ranking loss (https://github.com/youralien/skip-thoughts/blob/master/eval_rank.py)
        cost_im = margin - (lim * ls).sum(axis=1) + (lim * lcs).sum(axis=1)
        cost_im = cost_im * (cost_im > 0.) # this is like the max(0, pairwise-ranking-loss)
        cost_im = cost_im.sum(0)

        cost_s = margin - (ls * lim).sum(axis=1) + (ls * lcim).sum(axis=1)
        cost_s = cost_s * (cost_s > 0.) # this is like max(0, pairwise-ranking-loss)
        cost_s = cost_s.sum(0)

        cost = cost_im + cost_s
        # cost = PairwiseRanking(alpha=margin).apply(X, V, X_k, V_k)

        f = theano.function([lim, ls, lcim, lcs], cost, allow_input_downcast=True)


        print "\n each x, v in X,V are similar, cost should be low"
        X_tv = np.random.rand(*shape)
        V_tv = X_tv.copy()

        X_k_tv = np.random.rand(*shape)
        V_k_tv = np.random.rand(*shape)
        print "cost: ",f(X_tv, V_tv, X_k_tv, V_k_tv)

        print "\n each x, v in X,V are dissimlar, cost should be high"
        X_tv = np.random.rand(*shape)
        V_tv = np.random.rand(*shape)

        X_k_tv = np.random.rand(*shape)
        V_k_tv = np.random.rand(*shape)
        print "cost: ",f(X_tv, V_tv, X_k_tv, V_k_tv)

    test_cost()