import tensorflow as tf
from utils.tf_utils import fc
from tensorflow.contrib.layers import summarize_tensors


class FCModel(object):
    def __init__(self , obs_dim , acts_dim , lr=1e-2):
        self.obs_dim = obs_dim
        self.acts_dim = acts_dim
        self.global_step = tf.Variable(0 , trainable=False , name="global_step")
        self._init_ph()
        self._build_graph()
        self._train_op()
        self.sess = tf.Session()
        self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope='fc_model'))
        self.sess.run(tf.global_variables_initializer())
        self.summary = tf.summary.merge(summarize_tensors([self.loss]))


    def _init_ph(self):
        self.obs = tf.placeholder(tf.float32 , shape=(None , self.obs_dim))
        self.obs1 = tf.placeholder(tf.float32 , shape=(None , self.obs_dim))
        self.acts = tf.placeholder(tf.float32 , shape=(None , self.acts_dim))

    def _preprocess(self , act=tf.nn.elu):
        s = fc(self.obs , h_size=128 , act=tf.nn.elu , name='s')
        a = fc(self.acts , h_size=128 , act=tf.nn.elu , name='a')
        s1 = fc(self.obs1 , h_size=128 , act=tf.nn.elu , name='s1')
        return s , a , s1

    def _build_graph(self , n_layers=3):
        with tf.variable_scope('fc_model'):
            s , a , s1 = self._preprocess()
            x = tf.concat((s , a) , axis=1)
            h = fc(x , h_size=128 , name='fc' , act=tf.nn.tanh)
            for l in range(n_layers):
                h = fc(h , h_size=256 , name='fc_{}'.format(l) , act=tf.nn.tanh)
            self.s1_tilde = fc(h , h_size=self.obs_dim , name='s1_tilde' , act=None)

    def _train_op(self , lr=1e-4):
        self.loss = tf.reduce_mean(tf.square(self.s1_tilde - self.obs1))

        learning_rate = tf.train.exponential_decay(lr , self.global_step , 1024 * 3 , 0.96 ,
                                                   staircase=True)
        self.optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss ,
                                                                                  global_step=self.global_step)

    def train(self , obs , acts , obs1):
        _ , loss = self.sess.run([self.optim , self.loss] , feed_dict={
            self.obs: obs ,
            self.acts: acts ,
            self.obs1: obs1
        })

    def step(self , obs , acts):
        return self.sess.run(self.s1_tilde , feed_dict={self.obs: [obs] , self.acts: [acts]})

    def eval(self , obs , acts , obs1):
        loss = self.sess.run(self.loss ,
                             feed_dict={self.obs: obs ,
                                        self.acts: acts ,
                                        self.obs1: obs1})
        return loss
