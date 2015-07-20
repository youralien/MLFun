# # # # # # # # # # #
# Modeling Building #
# # # # # # # # # # #
from theano import tensor
import numpy as np
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.base import application, lazy
from blocks.bricks.recurrent import LSTM, GatedRecurrent, recurrent
from blocks.utils import shared_floatx
from blocks.bricks import Initializable, Linear
from blocks.bricks.lookup import LookupTable
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from blocks.monitoring import aggregation

class ShowAndTell(Initializable):

    def __init__(self, image_dim, dim, dictionary_size, max_sequence_length,
                 lookup_file=None, **kwargs):
        super(ShowAndTell, self).__init__(**kwargs)
        self.max_sequence_length = max_sequence_length

        # make image dimension of the embedding, so we can initialize
        # the hidden state with it
        image_embedding = Linear(
              input_dim=image_dim
            , output_dim=dim
            , name='image_embedding'
            )

        transition = GatedRecurrentWithInitialState(
            name='transition', dim=dim
            )

        if lookup_file:
            lookup = PretrainedLookupTable(lookup_file)
            feedback = PretrainedLookupFeedback(
                lookup=lookup, num_outputs=dictionary_size,feedback_dim=dim)
            print """
                  Warn: The pretrained lookup table you are supplying should
                  match the vectorizer loaded, which it was trained with.
                  """
        else:
            feedback = LookupFeedback(num_outputs=dictionary_size, feedback_dim=dim)

        readout = Readout(
              readout_dim=dictionary_size
            , source_names=["states"]
            , emitter=SoftmaxEmitter(name='emitter')
            , feedback_brick=feedback
            , name="readout"
            )

        generator = SequenceGenerator(
              readout=readout
            , transition=transition
            )

        self.image_embedding = image_embedding
        self.transition = transition
        self.generator = generator

        self.children = [
                  self.image_embedding
                , self.transition
                , self.generator
                , ]

    @application(inputs=["image_vects", "chars"], outputs=['out'])
    def cost(self, image_vects, chars):
        # shape (batch, features)
        image_embedding = self.image_embedding.apply(image_vects)

        cost = aggregation.mean(
              self.generator.cost_matrix(
                chars, cnn_context=image_embedding).sum()
            , chars.shape[1]
            )
        return cost

    @application
    def generate(self, image_vects):
        # shape (batch, features)
        image_embedding = self.image_embedding.apply(image_vects)
        return self.generator.generate(
                  n_steps=self.max_sequence_length
                , batch_size=image_embedding.shape[0]
                , iterate=True
                , cnn_context=image_embedding
                )

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

class GatedRecurrentWithInitialState(GatedRecurrent):
    
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, gate_activation=None,
                 **kwargs):
        super(GatedRecurrentWithInitialState, self).__init__(
            dim, activation, gate_activation, **kwargs)

    @recurrent(sequences=['mask', 'inputs', 'gate_inputs'],
               states=['states'], outputs=['states'],
               contexts=['cnn_context'])
    def apply(self, inputs, gate_inputs, states, mask=None, cnn_context=None):
        return super(GatedRecurrentWithInitialState, self).apply(
            inputs, gate_inputs, states, mask, iterate=False)

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        # cnn_context should be shape (batch_size, dim)
        cnn_context = kwargs['cnn_context']
        return [cnn_context]

class PretrainedLookupTable(LookupTable):
    def __init__(self, npy_file, **kwargs):
        super(PretrainedLookupTable, self).__init__(self)
        self.dim = 0
        self.length = 0
        self.npy_file = npy_file
        self.children = []

    def _allocate(self):
        data = np.load(self.npy_file)
        self.length, self.dim = data.shape
        self.parameters.append(
                shared_floatx(data, name="W"))

    def _initialize(self):
        pass

class PretrainedLookupFeedback(LookupFeedback):
    """A feedback brick for the case when readout are integers.
    Stores and retrieves distributed representations of integers.
    """
    def __init__(self, lookup, num_outputs, feedback_dim, **kwargs):
        super(PretrainedLookupFeedback, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.feedback_dim = feedback_dim

        self.lookup = lookup
        self.children = [self.lookup]

    @application
    def feedback(self, outputs):
        assert self.output_dim == 0
        return self.lookup.apply(outputs)

    def get_dim(self, name):
        return super(PretrainedLookupFeedback, self).get_dim(name)

# l2 norm, row-wise
def l2norm(X):
    norm = tensor.sqrt(tensor.pow(X, 2).sum(1))
    X /= norm[:, None]
    return X

if __name__ == '__main__':
    import theano
    import theano.tensor as T

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

        # image_vects_tv = np.zeros((batch_size, image_feature_dim), dtype='float32')
        word_vects_tv = np.zeros((batch_size, seq_len, embedding_dim), dtype='float32')
        # expecting sentence embedding to be [batch_size, embedding_dim]
        f = theano.function([word_vects], sem)
        print "input word vects shape: ", (batch_size, seq_len, embedding_dim)
        out = f(word_vects_tv)
        print "output shape"
        print out.shape
        print "output"
        print out

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
