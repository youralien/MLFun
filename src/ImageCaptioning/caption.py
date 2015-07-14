# common python
import decimal

# scientific python
import numpy as np
import theano
from theano import tensor as T

# blocks model building
from blocks.initialization import Uniform, IsotropicGaussian, Constant
from blocks.graph import ComputationGraph

# blocks model training
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.algorithms import GradientDescent, AdaDelta
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions import Printing, ProgressBar, FinishAfter
from blocks.extensions.training import TrackTheBest
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.saveload import Checkpoint

from fuel.transformers import Merge

# foxhound
from foxhound.preprocessing import Tokenizer
from foxhound import iterators
from foxhound.theano_utils import floatX, intX
from foxhound.transforms import SeqPadded

# local imports
from modelbuilding import Encoder, l2norm
from dataset import (coco, FoxyDataStream, GloveTransformer,
    ShuffleBatch, FoxyIterationScheme, loadFeaturesTargets, fillOutFilenames)
from utils import dict2json, json2dict, vStackMatrices, DecimalEncoder

# # # # # # # # # # #
# DataPreprocessing #
# # # # # # # # # # #
class DataETL():

    @staticmethod
    def getFinalStream(X, Y, sources, sources_k, batch_size=128, embedding_dim=300,
        min_df=2, max_features=50000):
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

        # vectorizer
        vect = Tokenizer(min_df=2, max_features=50000)
        vect.fit(trY)

        # Transforms
        trXt=lambda x: floatX(x)
        Yt=lambda y: intX(SeqPadded(vect.transform(y), 'back'))

        # Foxhound Iterators
        # RCL: Write own iterator to sample positive examples/captions, since there are 5 for each image.
        train_iterator = iterators.Linear(
            trXt=trXt, trYt=Yt, size=batch_size, shuffle=False
            )
        train_iterator_k = iterators.Linear(
            trXt=trXt, trYt=Yt, size=batch_size, shuffle=False
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

def train(
      sources = ("image_vects", "word_vects")
    , sources_k = ("image_vects_k", "word_vects_k")
    , batch_size=128
    , embedding_dim=300
    ):  
    trX, teX, trY, teY = coco(mode="full", batch_size=batch_size, n_captions=1)

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

    cg = ComputationGraph(cost)

    import ipdb
    ipdb.set_trace()
    # # # # # # # # # # #
    # Modeling Training #
    # # # # # # # # # # #

    algorithm = GradientDescent(
          cost=cost
        , parameters=cg.parameters
        , step_rule=AdaDelta()
        )
    main_loop = MainLoop(
          model=Model(cost)
        , data_stream=DataETL.getFinalStream(trX, trY, sources=sources, sources_k=sources_k, batch_size=batch_size)
        , algorithm=algorithm
        , extensions=[
              DataStreamMonitoring(
                  [cost]
                , DataETL.getFinalStream(trX, trY, sources=sources, sources_k=sources_k, batch_size=batch_size)
                , prefix='train')
            , DataStreamMonitoring(
                  [cost]
                , DataETL.getFinalStream(teX, teY, sources=sources, sources_k=sources_k, batch_size=batch_size)
                , prefix='test')
            # , ProgressBar()
            , Printing()
            # , FinishAfter(after_n_epochs=15)
            ]
        )
    main_loop.run()

    # ModelIO.save(f_emb, '/home/luke/datasets/coco/predict/fullencoder_maxfeatures.50000')
    ModelIO.save(f_emb, '/home/luke/datasets/coco/predict/encoder')

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
        Images: (z*N, K) matrix of images
        Captions: (z*N, K) matrix of captions
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

    foo()