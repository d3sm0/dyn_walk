import tensorflow as tf

from utils.tf_utils import fc, get_params

from tensorflow.contrib.layers import summarize_activation
class ValueNetwork(object):
    def __init__(self , name , obs_dim , h_size=(128 , 64 , 32) , act=tf.tanh):

        self.name = name
        with tf.variable_scope(name):
            self.init_ph(obs_dim)
            self.init_network(h_size , act)
            self._params = get_params(name)
            self.train_op()
            self.summarize = self.get_tensor_to_summarize()

    def train_op(self , lr=1e-3):
        # loss_1 = tf.square(self.vf - self.tdl)
        #
        # v_clipped = self.value_old + tf.clip_by_value(self.vf - self.value_old , -0.2 , 0.2)
        # loss_2 = tf.square(v_clipped - self.tdl)
        # self.loss = .5 * tf.reduce_mean(
        #     tf.maximum(loss_1 , loss_2))  # we do the same clipping-based trust region for the value function
        # print('Using trust region for VF')

        self.loss = .5 * tf.reduce_mean(tf.square(self.vf - self.tdl) , name='value_loss')
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        self.train = opt.minimize(self.loss , var_list=self._params)


    def get_grads(self):
        return tf.gradients(self.vf , self._params, name = 'value_gradient')

    def init_ph(self , obs_dim):
        self.obs = tf.placeholder('float32' , shape=(None , obs_dim) , name='state')
        self.tdl = tf.placeholder('float32' , shape=(None) , name='tdl')
        # self.value_old = tf.placeholder('float32' , shape=(None) , name='value_old')

    def init_network(self , h_size=(128 , 64 , 32) , act=tf.nn.tanh):
        h = act(fc(self.obs , h_size[0] , name='input'))
        for i in range(len(h_size)):
            h = fc(h , h_size[i] , act=act, name='h{}'.format(i))
        vf = fc(h , 1 , 'vf')
        self.vf = tf.squeeze(vf)

    def get_tensor_to_summarize(self):
        return self._params + [self.loss] + [self.obs, self.tdl, self.vf] + self.get_grads()
