import numpy as np
import tensorflow as tf


def lrelu(x, alpha = 0.1):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def fc(x , h_size , name , act=None , init=ortho_init(1.0)):
    with tf.variable_scope(name):
        input_size = x.get_shape()[1]
        w = tf.get_variable('w' , (input_size , h_size) , initializer = init)
        b = tf.get_variable('b' , (h_size) , initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x , w) + b
        if act is not None:
            z = act(z)
        return z


class GaussianPD(object):
    def __init__(self , mu , logstd):
        self.mu = mu
        self.std = tf.exp(logstd)
        self.logstd = logstd
        self.act_dim = tf.to_float(tf.shape(mu)[-1])

    def neglogp(self , action):
        return 0.5 * tf.reduce_sum(tf.square((action - self.mu) / self.std) , axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(action)[-1]) \
               + tf.reduce_sum(self.logstd , axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e) , axis=-1)

    def kl(self , pi_other):
        return tf.reduce_sum(
            pi_other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mu - pi_other.mu)) / (
                2.0 * tf.square(pi_other.std)) - 0.5 , axis=-1)

    def logp(self , action):
        return -self.neglogp(action)

    def sample(self):
        return self.mu + tf.random_normal(tf.shape(self.mu)) * self.std


def get_params(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope=scope)
