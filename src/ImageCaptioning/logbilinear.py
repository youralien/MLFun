"""
a log bilinear language model for generating words
author: rlouie

found and edited to work by this gist
https://gist.github.com/gouwsmeister/4739251
"""

import numpy as np
import theano
import theano.tensor as T
import math

# load random data
V = 1000    # num vocab
K = 20      # embedding dimensionality
context_sz = 3
mbatch_sz=10
num_points = 50
data = np.random.randint(V, size=(num_points, context_sz))
numbatches = data.shape[0] / mbatch_sz
# x = theano.shared(value=data, name='x')
labels = np.random.randint(V, size=(num_points,))
# y = theano.shared(value=labels, name='y')
 
# generate random embeddings matrix
rng=np.random.RandomState()
R_val = np.asarray(rng.uniform(-0.01, 0.01, size=(V,K)), dtype=theano.config.floatX)
R = theano.shared(value=R_val, name='R')
 
# model parameters
# KxK matrix for each context-word-embedding resulting in (context_sz, K, K) tensor
C_val = np.asarray(rng.normal(0, math.sqrt(0.1), size=(context_sz, K, K)), dtype=theano.config.floatX)
C = theano.shared(value=C_val, name='C')
# bias vector
b_w_val = np.asarray(rng.normal(0, math.sqrt(0.1), size=(V,)), dtype=theano.config.floatX)
b_w = theano.shared(value=b_w_val, name="b_w")
 
index = T.lvector()
span_start = index*mbatch_sz
span_stop = index*(mbatch_sz+1)
 
# build model
x_batch = T.lmatrix()
y_batch = T.lvector()
embeddings_batch = R[x_batch.flatten()].flatten().reshape((mbatch_sz, context_sz*K))
output_multinoms = T.zeros((mbatch_sz,V))
for i in xrange(mbatch_sz):
    word_i = T.tensordot(C, embeddings_batch[i,:].reshape((context_sz,K)), axes=[[0,2],[0,1]])
    # this_prediction = T.nnet.softmax(T.dot(word_i,R) + b_w).dimshuffle(1, 0)
    this_prediction = (T.dot(R, word_i) + b_w)
    output_multinoms = T.set_subtensor(output_multinoms[i, :], this_prediction)
 
# generate cost function
cost = -T.mean(T.log(output_multinoms)[T.arange(y_batch.shape[0]),y_batch])

def sgd(cost, params, lr=0.05):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

params = [R, C, b_w]
updates = sgd(cost, params)
# updates = {}
# learning_r = 0.01
# for param in [R, C, b_w]:
#     gparam = T.grad(cost, param)
#     updates[param] = param - learning_r*gparam
 
# train model
train_model = theano.function([x_batch, y_batch], [cost],
        updates=updates)

# epochs
for i in xrange(15):
	print "epoch %d" % i
	for start, end in zip(range(0, num_points, mbatch_sz), range(mbatch_sz, num_points, mbatch_sz)):
		cost = train_model(data[start:end], labels[start:end])
		print cost
