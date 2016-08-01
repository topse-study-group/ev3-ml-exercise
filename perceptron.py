import environ

import tensorflow as tf
import numpy as np
import math
from qnet import Qnet

class Perceptron(Qnet):
    def __init__(self):
        num_actions = len(environ.actions)
        self.x = tf.placeholder(tf.float32, [None, environ.status_length])
        self.w = tf.Variable(
                     tf.truncated_normal([environ.status_length, num_actions],
                     stddev=1.0/math.sqrt(2.0)))
        self.b = tf.Variable(tf.zeros([num_actions]))
        self.q = tf.matmul(self.x, self.w) + self.b
        
        self.y_ = tf.placeholder(tf.float32, [None, num_actions])
        self.loss = tf.reduce_mean(tf.square(self.y_ - self.q))
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
