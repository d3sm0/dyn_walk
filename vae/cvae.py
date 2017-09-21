import tensorflow as tf
from tensorflow.contrib.layers import summarize_tensors
import numpy as np
from networks import encoder , decoder


class VAE:
    def __init__(self , obs_dim , acts_dim , latent_dim , batch_size):
        """
        Implementation of Variational Autoencoder (VAE) for  MNIST.
        Paper (Kingma & Welling): https://arxiv.org/abs/1312.6114.

        :param latent_dim: Dimension of latent space.
        :param batch_size: Number of data points per mini batch.
        :param encoder: function which encodes a batch of inputs to a
            parameterization of a diagonal Gaussian
        :param decoder: function which decodes a batch of samples from
            the latent space and returns the corresponding batch of images.
        """
        self.z_dim = latent_dim
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.acts_dim = acts_dim

        self._init_ph()
        self._build_graph()
        self._train_op()
        # start tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.summary = tf.summary.merge(summarize_tensors([self.loss , self.kl , self.z , self.mean , self.sd]))

    def _init_ph(self):
        self.obs = tf.placeholder(tf.float32 , shape=[None , self.obs_dim])
        self.obs1 = tf.placeholder(tf.float32 , shape=[None , self.obs_dim])
        self.acts = tf.placeholder(tf.float32 , shape=[None , self.acts_dim])

    def _sample(self):
        return self.mean + tf.random_normal(tf.shape(self.mean)) * self.sd

    def _build_graph(self):
        """
        Build tensorflow computational graph for VAE.
        x -> encode(x) -> latent parameterization & KL divergence ->
        z -> decode(z) -> distribution over x -> log likelihood ->
        total loss -> train step
        """
        with tf.variable_scope('vae'):
            # encode inputs (map to parameterization of diagonal Gaussian)

            x = tf.concat((self.obs , self.acts , self.obs1) , axis=1)
            self.mean , self.sd = encoder(x , self.z_dim)
            self.z = self._sample()

            # with tf.variable_scope('sampling'):
            #     self.mean = self.encoded
            #     self.logvar = tf.get_variable('logvar' , shape=(1 , self.z_dim) , initializer=tf.zeros_initializer)
            #
            #     # also calculate standard deviation for practical use
            #     self.stddev = tf.sqrt(tf.exp(self.logvar))
            #
            #     # sample from latent space
            #     epsilon = tf.random_normal([self.batch_size , self.z_dim])
            #     self.z = self.mean + self.stddev * epsilon

            # deecode batch

            z = tf.concat((self.z , self.obs , self.acts) , axis=1)
            # z = self.z
            self.decoded = decoder(z , self.obs_dim)

            # calculate KL divergence between approximate posterior q and prior p

    def _train_op(self):
        kl = self.d_kl(self.mean , self.sd)
        self.kl = tf.reduce_mean(kl)

        log_lk = self.log_lik_bernoulli(self.obs , self.decoded)

        self.loss = tf.reduce_mean(kl - log_lk)

        self.train = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)

    @staticmethod
    def d_kl(mu , sigma , eps=1e-8):
        """
        Calculates KL Divergence between q~N(mu, sigma^T * I) and p~N(0, I).
        q(z|x) is the approximate posterior over the latent variable z,
        and p(z) is the prior on z.

        :param mu: Mean of z under approximate posterior.
        :param sigma: Standard deviation of z
            under approximate posterior.
        :param eps: Small value to prevent log(0).
        :return: kl: KL Divergence between q(z|x) and p(z).
        """
        var = tf.square(sigma)
        kl = 0.5 * tf.reduce_sum(tf.square(mu) + var - 1. - tf.log(var + eps) , axis=1)
        return kl

    @staticmethod
    def log_lik_bernoulli(targets , outputs , eps=1e-8):
        """
        Calculates negative log likelihood -log(p(x|z)) of outputs,
        assuming a Bernoulli distribution.

        :param targets: MNIST images.
        :param outputs: Probability distribution over outputs.
        :return: log_like: -log(p(x|z)) (negative log likelihood)
        """
        log_like = tf.reduce_sum(targets * tf.log(outputs + eps)
                                 + (1. - targets) * tf.log((1. - outputs) + eps) , axis=1)
        return log_like

    def train(self , obs , acts , obs1):
        """
        Performs one mini batch update of parameters for both inference
        and generative networks.

        :param x: Mini batch of input data points.
        :return: loss: Total loss (KL + NLL) for mini batch.
        """
        _ , loss = self.sess.run([self.train , self.loss] ,
                                 feed_dict={self.obs: obs ,
                                            self.acts: acts ,
                                            self.obs1: obs1})
        return loss

    def get_pi_z(self , obs , acts):
        """
        Maps a data point x (i.e. an image) to mean in latent space.

        :return: mean: mu such that q(z|x) ~ N(mu, .).
        """

        return self.sess.run([self.mean , self.sd] , feed_dict={self.obs: obs , self.acts: acts})

    def get_z(self , obs , acts):
        return self.sess.run(self.z , feed_dict={self.obs , self.acts})

    def get_obs1(self , obs , acts):
        """
        Generate from the model a new point. Either given by the latent space or given by obs, ancd acts

        :param z: Point in latent space.
        :return: x: Corresponding image generated from z.
        """

        assert np.ndim(obs) > 1 and np.ndim(acts) > 1

        return self.sess.run(self.decoded , feed_dict={self.obs: obs , self.acts: acts})
