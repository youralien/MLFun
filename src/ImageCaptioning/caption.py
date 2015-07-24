# common python
import decimal
import operator
from collections import OrderedDict

# scientific python
import numpy as np
import pandas as pd
import theano
from theano import tensor as T
from picklable_itertools.extras import equizip
from sklearn.preprocessing import MultiLabelBinarizer

# blocks model building
from blocks.initialization import Uniform, IsotropicGaussian, Constant
from blocks.graph import ComputationGraph

# blocks model training
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent, Adam
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions import Printing
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.filter import VariableFilter
from blocks.search import BeamSearch

from fuel.transformers import Merge

# foxhound
from foxhound.preprocessing import Tokenizer
from foxhound import iterators
from foxhound.theano_utils import floatX, intX
from foxhound.transforms import SeqPadded
from foxhound.rng import py_rng

# local imports
from modelbuilding import Encoder, l2norm
from dataset import (coco, cocoXYFilenames, FoxyDataStream, GloveTransformer,
    ShuffleBatch, FoxyIterationScheme, loadFeaturesTargets, fillOutFilenames,
    sbuXYFilenames)
from utils import dict2json, vStackMatrices, DecimalEncoder
from cuboid.extensions import UserFunc

# # # # # # #
#  Model IO #
# # # # # # #

import cPickle as pkl

class ModelIO():

    @staticmethod
    def save(func, saveto):
        pkl.dump(func, open('%s.pkl'%saveto, 'wb'))

    @staticmethod
    def load(saveto):
        func = pkl.load(open('%s.pkl'%saveto, 'r'))
        return func

# # # # # # # # # # #
# DataPreprocessing #
# # # # # # # # # # #

def sampleCaptions(ymb, K=1):
    """ymb = minibatch of captions
    it samples K captions from the available list of n captions
    """
    sampled_captions = []
    for captions_of_an_img in ymb:
        sampled_captions.extend(py_rng.sample(captions_of_an_img, K))
    return sampled_captions

def concatCaptions(ymb, K=5):
    """ymb = minibatch of captions
    it concatenates the first K captions from the available list of n captions.
    While destorying some sentence order when concatenating, this is
    helpful when we want the presence of a token"""
    
    def joinListOfCaptions(listOfCaptions):
        return " ".join(listOfCaptions[:K])
    return map(joinListOfCaptions, ymb)

def prepVect(min_df=2, max_features=50000, n_captions=5, n_sbu=None,
             multilabel=False):
    print "prepping the Word Tokenizer..."
    _0, _1, trY, _3 = coco(mode='full', n_captions=n_captions)
    if n_sbu:
        _4, sbuY, _5 = sbuXYFilenames(n_sbu)
        trY.extend(sbuY)
    vect = Tokenizer(min_df=min_df, max_features=max_features)
    captions = sampleCaptions(trY, n_captions) 
    vect.fit(captions)
    if multilabel:
        mlb = MultiLabelBinarizer()
        mlb.fit(vect.transform(captions))
        return vect, mlb
    # if not multilabel:
    return vect

dataset_name = 'coco_train2014'

# global vectorizer
vect_name = 'tokenizer_%s' % dataset_name
mlb_name = 'mlb_%s' % dataset_name
# vect_name = 'tokenizer_reddit' # gloveglove
# vect_name = 'tokenizer_coco_train2014+sbu100000'
try:
    if mlb_name:
        mlb = ModelIO.load(mlb_name)
        print "MLB loaded from file"
    vect = ModelIO.load(vect_name)
    # vect = ModelIO.load('tokenizer_reddit') # gloveglove
    print "Tokenizer loaded from file."
except:
    if mlb_name:
        vect, mlb = prepVect(n_sbu=100000, n_captions=1, multilabel=True)
        ModelIO.save(vect, vect_name)
        ModelIO.save(mlb, mlb_name)
        print "Saved %s, %s for future use." % (vect_name, mlb_name)
    else:
        vect = prepVect(n_sbu=100000, n_captions=1)
        ModelIO.save(vect, vect_name)
        print "Saved %s for future use." % vect_name

class DataETL():

    @staticmethod
    def getFinalStream(X, Y, sources, sources_k, batch_size=128, embedding_dim=300,
        shuffle=False):
        """Despite horrible variable names, this method
        gives back the final stream for both train or test data

        batch_size:
        embedding_dim: for glove vects
        min_df and max_features: for Tokenizer

        Returns
        -------
        merged stream with sources = sources + sources_k
        """
        trX, trY = (X, Y)
        trX_k, trY_k = (X, Y)

        # Transforms
        trXt=lambda x: floatX(x)
        Yt=lambda y: intX(SeqPadded(vect.transform(sampleCaptions(y)), 'back'))

        # Foxhound Iterators
        # RCL: Write own iterator to sample positive examples/captions, since there are 5 for each image.
        train_iterator = iterators.Linear(
            trXt=trXt, trYt=Yt, size=batch_size, shuffle=shuffle
            )
        train_iterator_k = iterators.Linear(
            trXt=trXt, trYt=Yt, size=batch_size, shuffle=shuffle
            )

        # FoxyDataStreams
        train_stream = FoxyDataStream(
              (trX, trY)
            , sources
            , train_iterator
            , FoxyIterationScheme(len(trX), batch_size)
            )

        train_stream_k = FoxyDataStream(
              (trX_k, trY_k)
            , sources_k
            , train_iterator_k
            , FoxyIterationScheme(len(trX), batch_size)
            )
        glove_version = "glove.6B.%sd.txt.gz" % embedding_dim
        train_transformer = GloveTransformer(
            glove_version, data_stream=train_stream, vectorizer=vect
            )
        train_transformer_k = GloveTransformer(
            glove_version, data_stream=train_stream_k, vectorizer=vect
            )

        # Final Data Streams w/ contrastive examples
        final_train_stream = Merge(
              (train_transformer, ShuffleBatch(train_transformer_k))
            , sources + sources_k
            )
        final_train_stream.iteration_scheme = FoxyIterationScheme(len(trX), batch_size)

        return final_train_stream

    @staticmethod
    def getTokenizedStream(X, Y, sources,
            batch_size=128, embedding_dim=300):
        """getTokenizedStream returns data with images as cnn features
        and captions as tokenized words. NOT glove vectors.

        This is being used for the end2end implementation. See google's paper
        "Show and Tell: Generating Image Captions".  Thus we are not using
        contrastive examples either.
        """
        trX, trY = (X, Y)

        # Transforms
        trXt=lambda x: floatX(x)
        Yt=lambda y: intX(SeqPadded(vect.transform(sampleCaptions(y)), 'back'))

        # Foxhound Iterators
        # RCL: Write own iterator to sample positive examples/captions, since there are 5 for each image.
        train_iterator = iterators.Linear(
            trXt=trXt, trYt=Yt, size=batch_size, shuffle=False
            )

        # FoxyDataStreams
        train_stream = FoxyDataStream(
              (trX, trY)
            , sources
            , train_iterator
            , FoxyIterationScheme(len(trX), batch_size)
            )

        train_stream.iteration_scheme = FoxyIterationScheme(len(trX), batch_size)
        return train_stream

    @staticmethod
    def getMLBStream(X, Y, sources, batch_size=128, embedding_dim=300,
            shuffle=False):
        """Returns sources in the format
        (sources[0], sources[1]--MultiLabelBinarized)
        """
        trX, trY = (X, Y)

        # Transforms
        trXt=lambda x: floatX(x)
        Yt=lambda y: intX(mlb.transform(vect.transform(concatCaptions(y))))

        # Foxhound Iterators
        train_iterator = iterators.Linear(
            trXt=trXt, trYt=Yt, size=batch_size, shuffle=shuffle
            )

        # FoxyDataStreams
        train_stream = FoxyDataStream(
              (trX, trY)
            , sources
            , train_iterator
            , FoxyIterationScheme(len(trX), batch_size)
            )

        train_stream.iteration_scheme = FoxyIterationScheme(len(trX), batch_size)

        return train_stream

def trainencoder(
      sources = ("image_vects", "word_vects")
    , sources_k = ("image_vects_k", "word_vects_k")
    , batch_size=128
    , embedding_dim=300
    , n_captions=5
    , n_sbu=None
    ):
    # data should not be shuffled, as there is semantics in their placement
    trX, teX, trY, teY = coco(mode='full', batch_size=batch_size, n_captions=n_captions)

    # add SBU
    if n_sbu:
        sbuX, sbuY, _ = sbuXYFilenames(n_sbu)
        trX.extend(sbuX)
        trY.extend(sbuY)
    # # # # # # # # # # #
    # Modeling Building #
    # # # # # # # # # # #

    s = Encoder(
          image_feature_dim=4096
        , embedding_dim=embedding_dim
        , biases_init=Constant(0.)
        , weights_init=Uniform(width=0.08)
        )
    s.initialize()

    image_vects = T.matrix(sources[0]) # named to match the source name
    word_vects = T.tensor3(sources[1]) # named to match the source name
    image_vects_k = T.matrix(sources_k[0]) # named to match the contrastive source name
    word_vects_k = T.tensor3(sources_k[1]) # named to match the contrastive source name

    # image_vects.tag.test_value = np.zeros((2, 4096), dtype='float32')
    # word_vects.tag.test_value = np.zeros((2, 15, 50), dtype='float32')
    # image_vects_k.tag.test_value = np.zeros((2, 4096), dtype='float32')
    # word_vects_k.tag.test_value = np.zeros((2, 15, 50), dtype='float32')

    # learned image embedding, learned sentence embedding
    lim, ls = s.apply(image_vects, word_vects)

    # learned constrastive im embedding, learned contrastive s embedding
    lcim, lcs = s.apply(image_vects_k, word_vects_k)

    # l2norms
    lim = l2norm(lim)
    lcim = l2norm(lcim)
    ls = l2norm(ls)
    lcs = l2norm(lcs)

    margin = 0.2 # alpha term, should not be more than 1!

    # pairwise ranking loss (https://github.com/youralien/skip-thoughts/blob/master/eval_rank.py)
    cost_im = margin - (lim * ls).sum(axis=1) + (lim * lcs).sum(axis=1)
    cost_im = cost_im * (cost_im > 0.) # this is like the max(0, pairwise-ranking-loss)
    cost_im = cost_im.sum(0)

    cost_s = margin - (ls * lim).sum(axis=1) + (ls * lcim).sum(axis=1)
    cost_s = cost_s * (cost_s > 0.) # this is like max(0, pairwise-ranking-loss)
    cost_s = cost_s.sum(0)

    cost = cost_im + cost_s
    cost.name = "pairwise_ranking_loss"

    # function to produce embedding
    f_emb = theano.function([image_vects, word_vects], [lim, ls])

    if n_sbu:
        sbuname = "sbu.%d" % n_sbu
    else:
        sbuname = ''
    name = "%s+coco_encoder_lstm_dim.%s" % (sbuname, embedding_dim)
    savename = '/home/luke/datasets/coco/predict/%s' % name

    def save_function(self):
        ModelIO.save(f_emb, savename)
        print "Similarity Embedding function saved while training"

    def rank_function(self):
        # Get 1000 images / captions to test rank
        stream = DataETL.getFinalStream(teX, teY, sources=sources,
                            sources_k=sources_k, batch_size=1000,
                            shuffle=True)
        
        images, captions, _0, _1 = stream.get_epoch_iterator().next()
        image_embs, caption_embs = f_emb(images, captions)
        ModelEval.ImageSentenceRanking(image_embs, caption_embs)

    cg = ComputationGraph(cost)

    # # # # # # # # # # #
    # Modeling Training #
    # # # # # # # # # # #

    algorithm = GradientDescent(
          cost=cost
        , parameters=cg.parameters
        , step_rule=Adam(learning_rate=0.0002)
        )
    main_loop = MainLoop(
          model=Model(cost)
        , data_stream=DataETL.getFinalStream(trX, trY, sources=sources,
              sources_k=sources_k, batch_size=batch_size)
        , algorithm=algorithm
        , extensions=[
              DataStreamMonitoring(
                  [cost]
                , DataETL.getFinalStream(trX, trY, sources=sources,
                      sources_k=sources_k, batch_size=batch_size)
                , prefix='train')
            , DataStreamMonitoring(
                  [cost]
                , DataETL.getFinalStream(teX, teY, sources=sources,
                      sources_k=sources_k, batch_size=batch_size)
                , prefix='test')
            , UserFunc(save_function, after_epoch=True)
            , UserFunc(rank_function, after_epoch=True)
            , Printing()
            , FinishIfNoImprovementAfter(notification_name="test_pairwise_ranking_loss",
                iterations=500)
            ]
        )
    main_loop.run()

    # ModelIO.save(f_emb, '/home/luke/datasets/coco/predict/fullencoder_maxfeatures.50000_epochsampler')
    ModelIO.save(f_emb, savename)

def trainend2end(
      sources = ("image_vects", "word_tokens")
    , batch_size=128
    , embedding_dim=300
    , n_captions=5
    , mode='sample'
    , n_sbu=None
    , recurrent_unit='lstm'
    ):
    """Train a full end to end system, w/out the encoder/decoder model.
    Like the google paper, "Show and Tell: Image Caption Generation"

    Like how we did it with MNIST
    """
    # data should not be shuffled, as there is semantics in their placement
    trX, teX, trY, teY = coco(mode="full", batch_size=batch_size, n_captions=n_captions)

    # add SBU
    if n_sbu:
        sbuX, sbuY, _ = sbuXYFilenames(n_sbu)
        trX.extend(sbuX)
        trY.extend(sbuY)


    image_vects = T.matrix(sources[0])
    word_tokens = T.lmatrix(sources[1])
    image_vects.tag.test_value = np.zeros((2, 4096), dtype='float32')
    word_tokens.tag.test_value = np.zeros((2, 15), dtype='int64')

    from modelbuilding import ShowAndTell
    show_and_tell = ShowAndTell(
          image_dim=4096
        , dim=embedding_dim
        , dictionary_size=vect.n_features
        , max_sequence_length=30
        # , lookup_file='glove_lookup_53454.npy' # gloveglove
        , recurrent_unit=recurrent_unit
        , norm=True
        , biases_init=Constant(0.)
        , weights_init=IsotropicGaussian(0.02)
        )
    show_and_tell.initialize()
    cost = show_and_tell.cost(image_vects, word_tokens)
    cost.name = "seq_log_likelihood"
    cg = ComputationGraph(cost)

    name = "sbu+coco_NIC_%s_dim.%s" % (show_and_tell.recurrent_unit, embedding_dim)
    savename = '/home/luke/datasets/coco/predict/%s' % name

    def save_f_gen(self):
        generated = show_and_tell.generate(image_vects)
        f_gen = ComputationGraph(generated).get_theano_function()
        ModelIO.save(f_gen, savename)
        print "Generation function saved while training"

    model = Model(cost)
    algorithm = GradientDescent(
          cost=cost
        , parameters=cg.parameters
        , step_rule=Adam(learning_rate=0.0002)
        )
    main_loop = MainLoop(
          model=model
        , data_stream=DataETL.getTokenizedStream(trX, trY, sources=sources,
            batch_size=batch_size)
        , algorithm=algorithm
        , extensions=[
              DataStreamMonitoring(
                    [cost]
                  , DataETL.getTokenizedStream(trX, trY, sources=sources,
                      batch_size=batch_size)
                  , prefix='train')
            , DataStreamMonitoring(
                    [cost]
                  , DataETL.getTokenizedStream(teX, teY, sources=sources,
                      batch_size=batch_size)
                  , prefix='test')
            , Printing()
            , UserFunc(save_f_gen, after_epoch=True)
            , FinishIfNoImprovementAfter(notification_name="test_seq_log_likelihood",
                iterations=1000)
            ]
        )
    main_loop.run()

    # Training finished; save the generator function w/ learned params
    generated = show_and_tell.generate(image_vects)

    # Beam Search
    if mode == "beam":
        samples, = VariableFilter(
            applications=[show_and_tell.generator.generate], name="outputs")(
                ComputationGraph(generated)) # generated[1] is next_outputs
        beam_search = BeamSearch(samples)
        try:
            path = '/home/luke/datasets/coco/predict/'
            filename = 'end2end_beam_maxseqlen.30_embeddingdim.300'
            ModelIO.save(beam_search, '%s%s' % (path, filename))
            print "It saved! Thanks pickle!"
        except Exception, e:
            print "Fuck pickle and move on with your life :)"
            print e
        ModelEval.beamsearch(beam_search)
    else:
        f_gen = ComputationGraph(generated).get_theano_function()
        try:
            name = "NIC_%s_dim.%s" % (show_and_tell.recurrent_unit, embedding_dim)
            ModelIO.save(f_gen, '/home/luke/datasets/coco/predict/%s' % name)
            print "It saved! Thanks pickle!"
        except Exception, e:
            print "Fuck pickle and move on with your life :)"
            print e
        ModelEval.predict(f_gen)

def sampleend2end(mode='sample', n_attempts=5):
    print "Loading Model..."
    if mode == "beam":
        path = '/home/luke/datasets/coco/predict/'
        filename = 'end2end_beam_maxseqlen.30_embeddingdim.300'
        beam_search = ModelIO.load('%s%s' % (path, filename))
        savepath = '/home/luke/datasets/coco/predict/generate_captions.json'
        ModelEval.beamsearch(beam_search, savepath=savepath, n_attempts=n_attempts)
    else:
        path = "/home/luke/datasets/coco/predict/"
        filename = 'sbu+coco_NIC_lstm_dim.300'
        f_gen = ModelIO.load(path + filename)
        savepath = '/home/luke/datasets/coco/predict/generate_captions.json'
        ModelEval.predict(f_gen, savepath=savepath, n_attempts=n_attempts)

def traindecoder(
      sources = ("image_vects", "word_vects")
    , sources_k = ("image_vects_k", "word_vects_k")
    , batch_size=128
    , embedding_dim=300
    , n_captions=5
    ):
    # data should not be shuffled, as there is semantics in their placement
    trX, teX, trY, teY = coco(mode="dev", batch_size=batch_size, n_captions=n_captions)

    # # # # # # # # # # #
    # Modeling Building #
    # # # # # # # # # # #

    stream = DataETL.getFinalStream(trX, trY, sources=sources, sources_k=sources_k,
            batch_size=batch_size)
    batch = stream.get_epoch_iterator().next()
    f_emb = ModelIO.load('/home/luke/datasets/coco/predict/fullencoder_maxfeatures.50000')
    import ipdb
    ipdb.set_trace()

    # # # # # # # # # # #
    # Modeling Training #
    # # # # # # # # # # #

#    algorithm = GradientDescent(
#          cost=cost
#        , parameters=cg.parameters
#        , step_rule=Adam()
#        )
#    main_loop = MainLoop(
#          model=Model(cost)
#        , data_stream=DataETL.getFinalStream(trX, trY, sources=sources,
#              sources_k=sources_k, batch_size=batch_size, n_captions=n_captions)
#        , algorithm=algorithm
#        , extensions=[
#              DataStreamMonitoring(
#                  [cost]
#                , DataETL.getFinalStream(trX, trY, sources=sources,
#                      sources_k=sources_k, batch_size=batch_size,
#                      n_captions=n_captions)
#                , prefix='train')
#            , DataStreamMonitoring(
#                  [cost]
#                , DataETL.getFinalStream(teX, teY, sources=sources,
#                      sources_k=sources_k, batch_size=batch_size,
#                      n_captions=n_captions)
#                , prefix='test')
#            # , ProgressBar()
#            , Printing()
#            # , FinishAfter(after_n_epochs=15)
#            ]
#        )
#    main_loop.run()


# # # # # # # # # # #
# Model Evaluation  #
# # # # # # # # # # #

class ModelEval():

    @staticmethod
    def rankcaptions(filenames, top_n=5):
        # n_captions = top_n # the captions it ranks as highest should all be relevant
        n_captions = 1 # RCL: image caption mismatch when n_captions is not just one
        batch_size = 128
        image_features, captions = loadFeaturesTargets(filenames, 'val2014', n_captions=n_captions)
        stream = DataETL.getFinalStream(
              image_features
            , captions
            , ("image_vects", "word_vects")
            , ("image_vects_k", "word_vects_k")
            , batch_size=batch_size
            )

        f_emb = ModelIO.load('/home/luke/datasets/coco/predict/fullencoder_maxfeatures.50000')
        im_emb, s_emb = None, None
        print "Computing Image and Text Embeddings"
        for batch in stream.get_epoch_iterator():
            im_vects = batch[0]
            s_vects = batch[1]
            batch_im_emb, batch_s_emb = f_emb(im_vects, s_vects)
            im_emb = vStackMatrices(im_emb, batch_im_emb)
            s_emb = vStackMatrices(s_emb, batch_s_emb)

        # account for make sure theres matching fns for each of the n_captions
        image_fns = fillOutFilenames(filenames, n_captions=n_captions)

        print "Computing Cosine Distances and Ranking Captions"
        relevant_captions = ModelEval.getRelevantCaptions(
            im_emb, s_emb, image_fns, captions, z=n_captions, top_n=top_n
        )
        dict2json(relevant_captions, "rankcaptions_fullencoder_maxfeatures.50000.json", cls=DecimalEncoder)
        return relevant_captions

    @staticmethod
    def rankscores(final_train_stream, final_test_stream, f_emb):

        i2t = ModelEval.i2t
        train_ep = final_train_stream.get_epoch_iterator()
        test_ep = final_test_stream.get_epoch_iterator()

        train_metrics = []
        test_metrics = []
        for train_data, test_data in train_ep, test_ep:
            im_emb, s_emb = f_emb(*train_data)
            train_metrics.append(i2t(im_emb, s_emb))
            im_emb, s_emb = f_emb(*train_data)
            test_metrics.append(i2t(im_emb, s_emb))
        train_metrics = np.vstack(train_metrics)
        test_metrics = np.vstack(test_metrics)

        metric_names = ("r1", "r5", "r10", "med")
        print "\nMean Metric Scores:"
        for i, metric_name in enumerate(metric_names):
            for metrics in (train_metrics, test_metrics):
                print "%s: %d" % metric_name, np.mean(metrics[:, i])

        return train_metrics, test_metrics

    @staticmethod
    def i2t(images, captions, z=1, npts=None):
        """
        Taken from https://github.com/ryankiros/skip-thoughts/blob/master/eval_rank.py
        Images: (z*N, K) matrix of image embeddings
        Captions: (z*N, K) matrix of caption embeddings
        """
        if npts == None:
            npts = images.shape[0] / z
        index_list = []

        # Project captions
        for i in range(len(captions)):
            captions[i] /= np.linalg.norm(captions[i])

        ranks = np.zeros(npts)
        for index in range(npts):

            # Get query image
            im = images[z * index].reshape(1, images.shape[1])
            im /= np.linalg.norm(im)

            # Compute scores
            d = np.dot(im, captions.T).flatten()
            inds = np.argsort(d)[::-1]
            index_list.append(inds[0])

            # Score
            rank = 1e20
            for i in range(z*index, z*index + z, 1):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        # Compute metrics
        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        return (r1, r5, r10, medr)

    @staticmethod
    def t2i(images, captions, z=1, npts=None):
        """
        Taken from https://github.com/ryankiros/skip-thoughts/blob/master/eval_rank.py
        Images: (z*N, K) matrix of image embeddings
        Captions: (z*N, K) matrix of captions embeddings
        """
        if npts == None:
            npts = images.shape[0] / z
        ims = np.array([images[i] for i in range(0, len(images), z)])


        # Project images
        for i in range(len(ims)):
            ims[i] /= np.linalg.norm(ims[i])

        # Project captions
        for i in range(len(captions)):
            captions[i] /= np.linalg.norm(captions[i])

        ranks = np.zeros(z * npts)
        for index in range(npts):

            # Get query captions
            queries = captions[z*index : z*index + z]

            # Compute scores
            d = np.dot(queries, ims.T)
            inds = np.zeros(d.shape)
            for i in range(len(inds)):
                inds[i] = np.argsort(d[i])[::-1]
                ranks[z * index + i] = np.where(inds[i] == index)[0][0]

        # Compute metrics
        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        medr = np.floor(np.median(ranks)) + 1
        return (r1, r5, r10, medr)

    @staticmethod
    def ImageSentenceRanking(images, captions, z=1):
        """
        Print nicely formatted tables each iteration
        N = 1000 is commonly used.
        images: (N, K)
        captions: (N, K)
        z: number of images per caption (see i2t, t2i)
        """
        rank_labels = ('R @ 1', 'R @ 5', 'R @ 10', 'Med R')
        image_annotation = ModelEval.i2t(images, captions, z)
        image_search = ModelEval.t2i(images, captions, z)

        print pd.DataFrame(OrderedDict(zip(rank_labels, image_annotation)), 
            index=pd.Index(["Image Annotation"])).to_string()
        print pd.DataFrame(OrderedDict(zip(rank_labels, image_search)), 
            index=pd.Index(["Image Search    "])).to_string()

        return image_annotation, image_search

    @staticmethod
    def ImageSearchSingleCategory(ims, mlb_matrix, captions, category_key, thresh):
        """do a single category, like dog"""
        # project images
        for i in range(len(ims)):
            ims[i] /= np.linalg.norm(ims[i])
        
        # project single captions
        for i in range(len(captions)):
            captions[i] /= np.linalg.norm(captions[i])

        assert captions.shape[0] == 1

        sims = np.dot(captions, ims.T).flatten()
        
        found = []
        n_found = 0
        n_matches = 0
        for i in range(len(sims)):
            if sims[i] > thresh:
                n_found += 1
                found.append((i, sims[i]))

                # depends on category_key being a single integer
                if mlb_matrix[i][category_key]:
                    n_matches += 1

        print "n_found: ", n_found
        print "n_matches: ", n_matches
        return found

    @staticmethod
    def getRelevantCaptions(im_emb, s_emb, image_fns, caption_strings, top_n, z=1, npts=None):
        """
        parameters
        ----------
        Images: (z*N, K) matrix of im_emb
        Captions: (z*N, K) matrix of captions
        image_fns: the filenames of im_emb for each image vectors in the im_emb matrix
        captions_strings: the captions (as strings) for each sentence vector in captions matrix

        Returns
        -------
        relevant_captions: dictionary storing the top_n rank predictions for each image file

        looks like
        {
            ... , 
            filepath.npy: {
                captions: ["caption with ranking 1", ...]
                cos_sims: [0.9, 0.5, ...]
            },
            ...
        }
        """
        if npts == None:
            npts = im_emb.shape[0] / z

        relevant_captions = {}

        # Project captions
        for i in range(len(s_emb)):
            s_emb[i] /= np.linalg.norm(s_emb[i])

        for index in range(npts):

            # Get query image
            im = im_emb[z * index].reshape(1, im_emb.shape[1])
            im /= np.linalg.norm(im)

            # Compute scores
            d = np.dot(im, s_emb.T).flatten() # cosine distance
            inds = np.argsort(d)[::-1] # sort by highest cosine distance

            # build up relevant top_n captions
            image_fn = image_fns[index]
            top_inds = inds[:top_n]
            top_captions = [caption_strings[ind] for ind in top_inds]
            top_cos_sims = [decimal.Decimal(float(d[ind])) for ind in top_inds]

            relevant_captions[image_fn] = {
                  "captions": top_captions
                , "cos_sims": top_cos_sims
                }

        return relevant_captions

    @staticmethod
    def predict(f_gen, X=None, Y=None, filenames=None, savepath=None, n_attempts=None):
        if X is None or Y is None or filenames is None:
            X, Y, filenames = cocoXYFilenames(dataType='val2014')
        ep = DataETL.getTokenizedStream(
            X=X, Y=Y, sources=('X', 'Y'), batch_size=1).get_epoch_iterator()
        if savepath:
            generated_captions = {}

        for filename in filenames:
            try:
                # No good way to make sure the filename is matching
                im_vects, txt_enc = ep.next()
                txt = " ".join(vect.inverse_transform(txt_enc))
                print "\nTrying for: ", txt
                if n_attempts:
                    batch_size = n_attempts
                else:
                    message=("Number of attempts to generate correct text? ")
                    batch_size = int(input(message)) 

                # make this call compatible with GRUs and LSTMs
                # GRU - generated is tuple of 3 elements (states, outputs, costs) 
                # LSTM - generated is tuple of 4 elements (states, cells, outputs, costs) 
                generated = f_gen(
                        np.repeat(im_vects, batch_size, 0)
                        )
                outputs = generated[-2]
                costs = generated[-1]

                outputs = list(outputs.T)
                costs = list(costs.T)
                for i in range(len(outputs)):
                    outputs[i] = list(outputs[i])
                    try:
                        # 0 is my PAD character, via foxhound Tokenizer
                        true_length = outputs[i].index(0)
                    except ValueError:
                        # full sequence length
                        true_length = len(outputs[i])
                    outputs[i] = outputs[i][:true_length]
                    costs[i] = costs[i][:true_length].mean()
                messages = []
                for sample, cost in equizip(outputs, costs):
                    # vect.inverse_transform needs a shape (seq, 1) array
                    sample = np.array(sample).reshape(-1, 1)
                    message = "({0:0.3f}) ".format(cost)
                    message += " ".join(vect.inverse_transform(sample))
                    messages.append((cost, message))
                messages.sort(key=operator.itemgetter(0), reverse=True)

                # convert to decimal to be picklable
                messages = [(decimal.Decimal(float(cost)), message) for cost, message in messages] 
                for _, message in messages:
                    print(message)
                if savepath:
                    generated_captions[filename] = messages
            except:
                if savepath:
                    dict2json(generated_captions, savepath, cls=DecimalEncoder)
                return

    @staticmethod
    def beamsearch(beam_search, X=None, Y=None, filenames=None, savepath=None, n_attempts=None):
        if X is None or Y is None or filenames is None:
            X, Y, filenames = cocoXYFilenames(dataType='val2014')
        ep = DataETL.getTokenizedStream(
            X=X, Y=Y, sources=('X', 'Y'), batch_size=1).get_epoch_iterator()
        if savepath:
            generated_captions = {}

        for filename in filenames:
            try:
                # No good way to make sure the filename is matching
                im_vects, txt_enc = ep.next()
                txt = " ".join(vect.inverse_transform(txt_enc))
                print "\nTrying for: ", txt
                if n_attempts:
                    batch_size = n_attempts
                else:
                    message=("Beam Size? ")
                    batch_size = int(input(message)) 
                input_ = np.repeat(im_vects, batch_size, 0)
                outputs, costs = beam_search.search(
                          input_values={"cnn_context": input_}
                        , eol_symbol=0 # encoder["PAD"] = 0 from foxhound tokenizer
                        , max_length=30 # hardcoded from how we trained it
                        )
                messages = []
                for sample, cost in equizip(outputs, costs):
                    # vect.inverse_transform needs a shape (seq, 1) array
                    sample = np.array(sample).reshape(-1, 1)
                    message = "({0:0.3f}) ".format(cost)
                    message += " ".join(vect.inverse_transform(sample))
                    messages.append((cost, message))
                messages.sort(key=operator.itemgetter(0), reverse=True)

                # convert to decimal to be picklable
                messages = [(decimal.Decimal(float(cost)), message) for cost, message in messages] 
                for _, message in messages:
                    print(message)
                if savepath:
                    generated_captions[filename] = messages
            except KeyboardInterrupt:
                if savepath:
                    dict2json(generated_captions, savepath, cls=DecimalEncoder)
                return

if __name__ == '__main__':
    def test_samplerankcaptions():
        import os
        dataDir='/home/luke/datasets/coco'
        dataType='val2014'
        test_fns = os.listdir("%s/features/%s"%(dataDir, dataType))
        test_fns = test_fns[:128]
        n_captions=3

        image_features, captions = loadFeaturesTargets(test_fns, 'val2014', n_captions=n_captions)
        image_fns = fillOutFilenames(test_fns, n_captions=n_captions)

        # these premade ones are shape (128, 300)
        batch_size = 128
        im_emb = np.load('im_emb.npy')
        s_emb = np.load('s_emb.npy')

        # GAH not sure if the captions will match, but this is just a test
        relevant_captions = ModelEval.getRelevantCaptions(im_emb, s_emb, image_fns[:batch_size], captions[:batch_size], top_n=3)
        print relevant_captions
        import ipdb
        ipdb.set_trace()

    # test_samplerankcaptions()



    def foo():# val_fns
        import os
        dataDir='/home/luke/datasets/coco'
        dataType='val2014'
        test_fns = os.listdir("%s/features/%s"%(dataDir, dataType))
        # test_fns = test_fns[:128*3]

        ModelEval.rankcaptions(test_fns)

    # foo()
    # traindecoder()
    # trainencoder(n_sbu=100000)
    # trainend2end(batch_size=100, embedding_dim=300, mode='sample', n_sbu=100000, recurrent_unit='lstm')
    # sampleend2end(n_attempts=20, mode='sample')
