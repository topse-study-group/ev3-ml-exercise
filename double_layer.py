import environ

import tensorflow as tf
import numpy as np
import math
from qnet import Qnet

class Double(Qnet):
    def __init__(self):
        H1_UNITS = 512
        H2_UNITS = 512
        num_actions = len(environ.actions)
        self.x = tf.placeholder(tf.float32, [None, environ.status_length])

        self.w1 = tf.Variable(
                tf.truncated_normal([environ.status_length, H1_UNITS], stddev=1.0/math.sqrt(2.0)),
                name='weights')
        self.b1 = tf.Variable(tf.zeros([1, H1_UNITS]), name='biases')
        self.hidden1 = tf.nn.relu(tf.matmul(self.x, self.w1) + self.b1)

        self.w2 = tf.Variable(
            tf.truncated_normal([H1_UNITS, H2_UNITS], stddev=1.0/math.sqrt(2.0)),
            name='weights')
        self.b2 = tf.Variable(tf.zeros([1, H2_UNITS]), name='biases')
        self.hidden2 = tf.nn.relu(tf.matmul(self.hidden1, self.w2) + self.b2)
            
        self.w = tf.Variable(
            tf.truncated_normal([H2_UNITS, num_actions], stddev=1.0/math.sqrt(2.0)),
            name='weights')
        self.b = tf.Variable(tf.zeros([num_actions]), name='biases')
        self.q = tf.matmul(self.hidden2, self.w) + self.b

        self.y_ = tf.placeholder(tf.float32, [None, num_actions])
        self.loss = tf.reduce_mean(tf.square(self.y_ - self.q))
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def reset(self):
        self.sess.close()
        self.__init__()

