import tensorflow as tf

from utils.tf_utils import GaussianPD , fc


class PolicyNetwork(object):
    def __init__(self , name , obs_dim , act_dim , kl_target=0.003 , eta=50 , act=tf.tanh , h_size=(128 , 64 , 32)):
        with tf.variable_scope(name):
            # self.g = tf.Graph()
            # with self.g.as_default():
            self._init_ph(obs_dim , act_dim)
            self._init_network(act_dim , act=act , h_size=h_size)
            self._init_pi(act_dim)
            self.losses = self.train_op(kl_target , eta)

    def _init_ph(self , obs_dim , act_dim):
        self.obs = tf.placeholder('float32' , (None , obs_dim) , name='state')
        self.acts = tf.placeholder('float32' , (None , act_dim) , name='action')
        self.adv = tf.placeholder('float32' , (None) , name='advantage')
        self.beta = tf.placeholder('float32' , () , name='beta')
        self.lr = tf.placeholder('float32' , () , name='lr')
        self.mu_old = tf.placeholder('float32' , (None , act_dim) , name='mu_old')
        self.logstd_old = tf.placeholder('float32' , (1 , act_dim) , name='logstd_old')

    def _init_network(self , act_dim , h_size=(128 , 64 , 32) , act=tf.tanh):
        h = act(fc(self.obs , h_size[0] , name='input'))
        for i in range(len(h_size)):
            h = act(fc(h , h_size[i] , name='h{}'.format(i)))

        self.mu = fc(h , act_dim , name='mu')
        self.logstd = tf.get_variable('log_std' , (1 , act_dim) , tf.float32 , tf.zeros_initializer())
        self.std = tf.exp(self.logstd)

    def _init_pi(self , action_dim):
        # self.logpi = self._logpi(mu=self.mu , logstd=self.logstd , act_dim=action_dim)
        # self.logpi_old = self._logpi(mu=self.mu_old , logstd=self.logstd_old , act_dim=action_dim)

        self.pi = GaussianPD(self.mu , self.logstd)
        self.pi_old = GaussianPD(self.mu_old , self.logstd_old)
        # # compute D_KL [pi_old || pi]
        self.kl = tf.reduce_mean(self.pi_old.d_kl(self.pi))
        self.entropy = self.pi.entropy()[0]
        self.sample = self.pi.sample()

    def train_op(self , kl_target=0.003 , eta=1000 , lr=1e-4):
        # alternative loss from baseline ppo
        # #
        ratio = tf.exp(self.pi.logpi(self.acts) - self.pi_old.logpi(self.acts))
        loss_1 = ratio * self.adv
        loss_2 = tf.clip_by_value(ratio , 1 - 0.2 , 1 + 0.2) * self.adv
        # loss_3 = (-self.pi.entropy())

        # loss_1 = -tf.reduce_mean(self.adv * tf.exp(self.pi.logpi(self.acts) - self.pi_old.logpi(self.acts)))
        # loss_2 = tf.multiply(self.beta , self.kl)
        loss_3 = eta * tf.square(tf.maximum(0.0 , self.kl - 2.0 * kl_target))
        # self.loss = loss_1 + loss_3 + loss_2
        self.loss = -tf.reduce_mean(tf.minimum(loss_1 , loss_2)) + loss_3  # PPO pessimistic surrogate
        self.train = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        return (loss_1 , loss_2 , loss_3)

    def get_params(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES ,
                                 scope=tf.get_variable_scope().original_name_scope)

    def get_grads(self):
        return tf.gradients(self.loss , self.get_params())

        # def logpi(self , action):
        #     k = np.log(2.0 * np.pi) * self.act_dim
        #     neglogpi = 0.5 * tf.reduce_sum(tf.square(action - self.mu / self.std) , axis=-1) + 0.5 * k + tf.reduce_sum(
        #         self.logstd , axis=-1)
        #     logpi = -neglogpi
        #     return logpi
        #
        # def _logpi(self , mu , logstd , act_dim):
        #     k = np.log(2.0 * np.pi) * act_dim
        #     neglogpi = 0.5 * tf.reduce_sum(tf.square(self.acts - mu / tf.exp(logstd)) ,
        #                                    axis=-1) + 0.5 * k + tf.reduce_sum(
        #         logstd , axis=-1)
        #     logpi = -neglogpi
        #     return logpi
        #
        # def _entropy(self):
        #     return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e) , axis=-1)
        #
        # def d_kl(self , pi_other):
        #     return tf.reduce_sum(
        #         pi_other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mu - pi_other.mu)) / (
        #             2.0 * tf.square(pi_other.std)) - 0.5 , axis=-1)
        #
        # def _d_kl(self):
        #     return tf.reduce_sum(
        #         self.logstd - self.logstd_old + (tf.square(tf.exp(self.logstd_old)) + tf.square(self.mu_old - self.mu)) / (
        #             2.0 * tf.square(self.std)) - 0.5 , axis=-1)
        #
        # def sample(self):
        #     return self.mu + tf.random_normal(tf.shape(self.mu)) * self.std
        #
        # def logpi(self , action):
        #     k = np.log(2.0 * np.pi) * self.act_dim
        #     neglogpi = 0.5 * tf.reduce_sum(tf.square(action - self.mu / self.std) , axis=-1) + 0.5 * k + tf.reduce_sum(
        #         self.logstd , axis=-1)
        #     logpi = -neglogpi
        #     return logpi
        #
        # def entropy(self):
        #     return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e) , axis=-1)



        # def sync_op(self):
        #
        #     assert len(self.params) == len(self.pi_old.params)
        #
        #     params = []
        #     # transfer new policy value to old policy
        #     for (pi_old , pi) in zip(self.pi_old.params , self.params):
        #         params.append(tf.assign(pi_old , pi))
        #     return params
