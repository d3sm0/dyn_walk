import tensorflow as tf
from tf_utils import fc


class ValueNetwork(object):
    def __init__(self , name, obs_dim , h_size = 128 , act=tf.tanh):

        with tf.variable_scope(name):
            self.init_ph(obs_dim)
            self.init_network(h_size , act)
            self.var_list()
            self.grad_op()
            self.train_op()

    def train_op(self):
        self.loss = tf.reduce_mean(tf.square(self.vf - self.tdl) , name='value_loss')
        self.train = tf.train.AdamOptimizer().minimize(self.loss)

    def var_list(self):
        self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES , scope=tf.get_variable_scope().name)

    def grad_op(self):
        self.grads = tf.gradients(self.vf , self.params)

    def init_ph(self , obs_dim):
        self.obs = tf.placeholder('float32' , shape=(None , obs_dim) , name='state')
        self.tdl = tf.placeholder('float32' , shape=(None ,) , name='tdl')

    def init_network(self , h_size , act):
        h0 = act(fc(self.obs , h_size , 'h0'))
        h1 = act(fc(h0 , h_size , 'h1'))
        h2 = act(fc(h1 , 64 , 'h2'))
        vf = fc(h2 , 1 , 'vf')
        self.vf = tf.squeeze(vf)
