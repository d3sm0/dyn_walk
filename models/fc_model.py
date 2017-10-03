import tensorflow as tf
from tensorflow.contrib.layers import summarize_tensors

from utils.tf_utils import fc
from kafnets.tf_kafnets import KAFNet
from scipy import spatial
import numpy as np


class FCModel(object):
    def __init__(self, obs_dim, acts_dim, is_recurrent=False, lr=1e-2):
        self.obs_dim = obs_dim
        self.acts_dim = acts_dim
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self._init_ph()
        self._build_graph(is_recurrent=is_recurrent)
        self._train_op()
        self.sess = tf.Session()
        self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_model'))
        self.sess.run(tf.global_variables_initializer())
        self.summary = tf.summary.merge(summarize_tensors([self.img_loss]))

    def _init_ph(self):
        self.obs = tf.placeholder(tf.float32, shape=(None, self.obs_dim))
        self.obs1 = tf.placeholder(tf.float32, shape=(None, self.obs_dim))
        self.acts = tf.placeholder(tf.float32, shape=(None, self.acts_dim))

    def _preprocess(self, act=tf.nn.elu):
        s = self.obs
        a = self.acts
        s1 = self.obs1
        return s, a, s1

    def _build_graph(self, is_recurrent=False, n_layers=3, act = tf.nn.tanh, dict_size = 10):
        with tf.variable_scope('fc_model'):
            s, a, s1 = self._preprocess()
            x = tf.concat((s, a), axis=1)
            d = tf.linspace(start=-2., stop=2., num=dict_size)

            h = fc(x, h_size=128, name='fc', act=act)

            # alpha = tf.get_variable('fc_a', shape=(128, dict_size))
            # h = KAFNet.kaf(linear=h, D=d, alpha=alpha, kernel='rbf')
            if is_recurrent:
                from tensorflow.contrib.rnn import BasicLSTMCell
                cell = BasicLSTMCell(num_units=32)
                h, _ = tf.nn.dynamic_rnn(cell=cell, inputs=tf.expand_dims(h, [0]), time_major=False, dtype=tf.float32)
                h = tf.reshape(h, (-1, cell.state_size.c))
            else:
                for l in range(n_layers):
                    h = fc(h, h_size=128, name='fc_{}'.format(l), act=act)
                    # alpha = tf.get_variable('fc_{}'.format(l), shape=(128, dict_size))
                    # h = KAFNet.kaf(kernel='periodic', linear=h, D=d, alpha=alpha)

            self.s1_tilde = fc(h, h_size=self.obs_dim, name='s1_tilde', act=None)
            self.conf_tilde = fc(h, h_size=1, name='conf', act=None)

    def _train_op(self, lr=1e-4):
        self.img_loss = tf.reduce_mean(tf.square(self.s1_tilde - self.obs1))
        self.conf_loss = tf.reduce_mean(tf.square(self.conf_tilde - self.img_loss))

        self.train_conf = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.conf_loss, var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_model/conf'))

        self.train_img = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.img_loss, global_step=self.global_step)

    def train(self, obs, acts, obs1, m=0):
        if m == 0:
            fetches = [self.train_img, self.img_loss]
        else:
            fetches = [self.train_conf, self.conf_loss]

        _, loss = self.sess.run(fetches, feed_dict={
            self.obs: obs,
            self.acts: acts,
            self.obs1: obs1
        })
        return loss

    def step(self, obs, acts):
        return self.sess.run(self.s1_tilde, feed_dict={self.obs: [obs], self.acts: [acts]})

    def eval(self, obs, acts, obs1):
        loss = self.sess.run(self.img_loss,
                             feed_dict={self.obs: obs,
                                        self.acts: acts,
                                        self.obs1: obs1})
        return loss

    def confidence(self, obs, acts):
        conf_tilde = self.sess.run(self.conf_tilde,
                                   feed_dict={self.obs: [obs],
                                              self.acts: [acts]})
        return 1 - abs(conf_tilde)
