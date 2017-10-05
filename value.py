import tensorflow as tf
from kafnets.tf_kafnets import KAFNet

from utils.tf_utils import fc, get_params, _clip_grad


class ValueNetwork(object):
    def __init__(self, name, obs_dim, h_size=(128, 64, 32), act=tf.tanh, dict_size = 20):
        self.name = name
        with tf.variable_scope(name):
            self.init_ph(obs_dim)
            self._init_network(h_size, act, dict_size=dict_size)
            self._params = get_params(name)
            self.train_op()
            self.summarize = self.get_tensor_to_summarize()

    def train_op(self, lr=1e-3):
        self.loss = tf.reduce_mean(tf.square(self.vf - self.tdl), name='value_loss')
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        gvs = _clip_grad(self.loss, self._params)
        self.train = opt.apply_gradients(gvs)

    def get_grads(self):
        return tf.gradients(self.vf, self._params, name='value_gradient')

    def init_ph(self, obs_dim):
        self.obs = tf.placeholder('float32', shape=(None, obs_dim), name='state')
        self.tdl = tf.placeholder('float32', shape=(None), name='tdl')

    def _init_network(self, h_size=(128, 64, 32), act=tf.nn.tanh, dict_min = -2., dict_max = 2., dict_size = 20):
        h = fc(self.obs, h_size[0], name='input')

        if act == 'kafnet':
            d = tf.linspace(start=dict_min, stop=dict_max, num=dict_size)
            alpha = tf.get_variable('fc_a', shape=(h_size[0], dict_size))
            h = KAFNet.kaf(linear=h, D=d, alpha=alpha)
        else:
            h = act(h)

        for i in range(len(h_size)):
            h = fc(h, h_size[i], act=None, name='h{}'.format(i))
            if act == 'kafnet':
                alpha = tf.get_variable(name='h{}'.format(i), shape=(h_size[i], dict_size))
                h = KAFNet.kaf(linear=h, D=d, alpha=alpha, kernel='rbf')
            else:
                h = act(h)
        vf = fc(h, 1, 'vf')
        self.vf = tf.squeeze(vf)

    def get_tensor_to_summarize(self):
        return self._params + [self.loss] + [self.obs, self.tdl, self.vf] + self.get_grads()
