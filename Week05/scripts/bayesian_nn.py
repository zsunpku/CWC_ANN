#!/usr/bin/env python
"""Bayesian neural network using variational inference
(see, e.g., Blundell et al. (2015); Kucukelbir et al. (2016)).

Inspired by autograd's Bayesian neural network example.
This example prettifies some of the tensor naming for visualization in
TensorBoard. To view TensorBoard, run `tensorboard --logdir=log`.


Modified by A. Massmann for CWC ANN meeting, orignal copy avail.:
https://github.com/blei-lab/edward/blob/master/examples/bayesian_nn.py
References
----------
http://edwardlib.org/tutorials/bayesian-neural-network
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf
import time
from edward.models import Normal, Categorical

import keras
from keras.datasets import mnist # load up the training data!

D = 28*28   # number of features

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype(np.float32)/255.
x_test = x_test.astype(np.float32)/255.
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

ed.set_seed(42)

def neural_network(x, W_0, W_1, W_2, b_0, b_1, b_2):
  """define NN using hyperbolic tan functions"""
  h = tf.tanh(tf.matmul(x, W_0) + b_0)
  h = tf.tanh(tf.matmul(h, W_1) + b_1)
  h = tf.matmul(h, W_2) + b_2
  return h #tf.reshape(h, [-1])

N = len(y_train)

ed.set_seed(42)

# MODEL
with tf.name_scope("model"):
  W_0 = Normal(loc=tf.zeros([D, 10]), scale=tf.ones([D, 10]), name="W_0")
  W_1 = Normal(loc=tf.zeros([10, 10]), scale=tf.ones([10, 10]), name="W_1")
  W_2 = Normal(loc=tf.zeros([10, 10]), scale=tf.ones([10, 1]), name="W_2")
  b_0 = Normal(loc=tf.zeros(10), scale=tf.ones(10), name="b_0")
  b_1 = Normal(loc=tf.zeros(10), scale=tf.ones(10), name="b_1")
  b_2 = Normal(loc=tf.zeros(10), scale=tf.ones(10), name="b_2")

  x = tf.placeholder(tf.float32, [N, D], name="X")
  y = Categorical(logits=neural_network(x, W_0, W_1, W_2, b_0, b_1, b_2),\
                  dtype=tf.int64)


# INFERENCE
with tf.name_scope("posterior"):
  with tf.name_scope("qW_0"):
    qW_0 = Normal(loc=tf.Variable(tf.random_normal([D, 10]), name="loc"),
                  scale=tf.nn.softplus(
                      tf.Variable(tf.random_normal([D, 10]), name="scale")))
  with tf.name_scope("qW_1"):
    qW_1 = Normal(loc=tf.Variable(tf.random_normal([10, 10]), name="loc"),
                  scale=tf.nn.softplus(
                      tf.Variable(tf.random_normal([10, 10]), name="scale")))
  with tf.name_scope("qW_2"):
    qW_2 = Normal(loc=tf.Variable(tf.random_normal([10, 10]), name="loc"),
                  scale=tf.nn.softplus(
                      tf.Variable(tf.random_normal([10, 10]), name="scale")))
  with tf.name_scope("qb_0"):
    qb_0 = Normal(loc=tf.Variable(tf.random_normal([10]), name="loc"),
                  scale=tf.nn.softplus(
                      tf.Variable(tf.random_normal([10]), name="scale")))
  with tf.name_scope("qb_1"):
    qb_1 = Normal(loc=tf.Variable(tf.random_normal([10]), name="loc"),
                  scale=tf.nn.softplus(
                      tf.Variable(tf.random_normal([10]), name="scale")))
  with tf.name_scope("qb_2"):
    qb_2 = Normal(loc=tf.Variable(tf.random_normal([10]), name="loc"),
                  scale=tf.nn.softplus(
                      tf.Variable(tf.random_normal([10]), name="scale")))

# hack to get score/performance
x_test = np.vstack([x_test]*6)
y_test = np.hstack([y_test]*6)

y_sample = tf.stack(
  [Categorical(logits=neural_network(x, qW_0.sample(),\
                                     qW_1.sample(), qW_2.sample(),\
                                     qb_0.sample(), qb_1.sample(),\
                                     qb_2.sample()))
               for _ in range(10)])

# build graph
inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                     W_1: qW_1, b_1: qb_1,
                     W_2: qW_2, b_2: qb_2}, data={x: x_train, y: y_train})

# fit model
start = time.time()
inference.run(n_iter=1000, n_samples=1, logdir='log')
end = time.time()
print('training wall clock time: %5.2f min' % ((end-start)/60.))


sampled_y = y_sample.eval({x: x_test})
accuracy = np.ones(sampled_y.shape[0])*np.nan
for i,_y in enumerate(sampled_y):
  error = _y - y_test
  accuracy[i] = float(error[error == 0].shape[0])\
                  /float(error.shape[0])

print('mean accuracy: %f' % accuracy.mean())
print('std accuracy: %f' % accuracy.std())
