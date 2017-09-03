import tensorflow as tf
from tf_utils import GaussianPD , fc


# TODO: Class inheritance between pi and pi old

class PolicyNetwork(object):
    def __init__(self , name , obs_dim , act_dim , kl_target=0.003 , pi_old=None , eta=50 , act=tf.tanh , h_size=128):
        with tf.variable_scope(name):
            self.init_ph(obs_dim , act_dim)
            self.init_network(act_dim , act=act , h_size=h_size)
            self.init_pi()
            self.var_list()
            if pi_old is not None:
                self.pi_old = pi_old
                self.kl = tf.reduce_mean(self.pi_old.pi.d_kl(self.pi))
                self.sync = self.sync_op()
                self.losses = self.train_op(kl_target , eta)
                self.grad_op()

    def init_ph(self , obs_dim , act_dim):
        self.obs = tf.placeholder('float32' , (None , obs_dim) , name='state')
        self.acts = tf.placeholder('float32' , (None , act_dim) , name='action')
        self.adv = tf.placeholder('float32' , (None) , name='advantage')
        self.beta = tf.placeholder('float32' , () , name='beta')
        self.lr = tf.placeholder('float32', (), name  = 'lr')
        self.mu_old = tf.placeholder('float32' , (None , act_dim) , name='mu_old')
        self.logstd_old = tf.placeholder('float32' , (1 , act_dim) , name='logstd_old')

    def init_network(self , act_dim , n_layer=2 , h_size=128 , act=tf.tanh):
        h0 = act(fc(self.obs , h_size , name='h_0'))
        h1 = act(fc(h0 , h_size , name='h1'))
        h2 = act(fc(h1 , 64 , name='h2'))
        h3 = act(fc(h2 , 32 , name='h3'))
        self.mu = fc(h3 , act_dim , name='mu')
        self.logstd = tf.get_variable('log_std' , (1 , act_dim) , tf.float32 , tf.zeros_initializer())

    def init_pi(self):
        self.pi = GaussianPD(self.mu , self.logstd)
        # self.pi_old = GaussianPD(self.mu_old , self.logstd_old)
        # # compute D_KL [pi_old || pi]
        # self.kl = self.pi_old.d_kl(self.pi)

    def train_op(self , kl_target=0.003 , eta=50 , lr=1e-4):

        # alternative loss from baseline ppo
        # ratio = tf.exp(self.pi.logpi(self.acts) - self.pi_old.pi.logpi(self.acts))
        # loss_1 = ratio*self.adv
        # loss_2 = tf.clip_by_value(ratio, 1-0.2, 1+0.2)*self.adv
        # loss = -tf.reduce_mean(tf.minimum(loss_1, loss_2)) # PPO pessimistic surrogate
        # # loss_3 = (-entcoeff) * meanent


        loss_1 = -tf.reduce_mean(self.adv * tf.exp(self.pi.logpi(self.acts) - self.pi_old.pi.logpi(self.acts)))
        loss_2 = tf.multiply(self.beta , self.kl)
        loss_3 = eta * tf.square(tf.maximum(0.0 , self.kl - 2.0 * kl_target))
        self.loss = loss_1 + loss_3 + loss_2
        self.train = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        return (loss_1 , loss_2 , loss_3)

    def var_list(self):
        self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES ,
                                        scope=tf.get_variable_scope().original_name_scope)

    def grad_op(self):
        self.grads = tf.gradients(self.loss , self.params)

    def sync_op(self):

        assert len(self.params) == len(self.pi_old.params)

        params = []
        # transfer new policy value to old policy
        for (pi_old , pi) in zip(self.pi_old.params , self.params):
            params.append(tf.assign(pi_old , pi))
        return params
