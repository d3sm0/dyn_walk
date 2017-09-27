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



from tensorflow.contrib.rnn import BasicLSTMCell

def recurrent_encoder(x , z_dim , act=tf.nn.elu):
    with tf.variable_scope('encoder'):
        # e = fully_connected(x , 256 , scope='fc-01' , activation_fn=act)
        # e = fully_connected(e , 128 , scope='fc-03' , activation_fn=act)
        cell = BasicLSTMCell(num_units=64)
        e,state_out = tf.nn.dynamic_rnn(cell=cell , inputs=tf.expand_dims(x , axis=0) , time_major=False , dtype=tf.float32)
        e = tf.reshape(e , (-1 , cell.state_size.c))
        e = fc(e , 64 , name='fc-04' , act=act)
        # e = fully_connected(e, 2 * latent_dim, activation_fn=None,
        #                            scope='fc-final')
        mean = fc(e , z_dim , name = 'mean', act=None)
        logsd = tf.get_variable('logsd' , shape=(1 , z_dim) , initializer=tf.zeros_initializer)
    return mean , logsd


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
