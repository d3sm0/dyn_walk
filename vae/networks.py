import tensorflow as tf

from utils.tf_utils import fc


def encoder(x , z_dim , act=tf.nn.elu):
    """
    Inference network q(z|x) which encodes a mini batch of data points
    to a parameterization of a diagonal Gaussian using a network with 
    fully connected layers.

    :param x: Mini batch of data points to encode.
    :param latent_dim: dimension of latent space into which we encode
    :return: e: Encoded mini batch.
    """

    with tf.variable_scope('encoder'):
        h = fc(x , 256 , name='h' , act=act)
        h1 = fc(h , 128 , name='h1' , act=act)
        h2 = fc(h1 , 64 , name='h2' , act=act)
        # e = fc(e, 2 * latent_dim, act=None,
        #                            name='fc-final')
        mean = fc(h2 , z_dim , name='mean' , act=None)
        logsd = tf.get_variable('logsd' , shape=(1 , z_dim) , initializer=tf.zeros_initializer)
    return mean , tf.exp(logsd)


def decoder(z , obs_dim , act=tf.nn.elu):
    """
    Generative network p(x|z) which decodes a sample z from
    the latent space using a network with fully connected layers.

    :param z: Latent variable sampled from latent space.
    :return: x: Decoded latent variable.
    """
    with tf.variable_scope('decoder'):
        h2 = fc(z , 128 , name='fc-01' , act=act)
        h1 = fc(h2 , 128 , name='fc-02' , act=act)
        h = fc(h1 , 256 , name='fc-03' , act=act)
        obs_1_tilde = fc(h , obs_dim , act=tf.nn.sigmoid , name='fc-final')

    return obs_1_tilde
