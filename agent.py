from value import ValueNetwork
from policy import PolicyNetwork
import tensorflow as tf
import numpy as np


class Agent(object):
    def __init__(self , obs_dim , act_dim , kl_target=1e-2 , eta=1000 , beta=1.0 , h_size=128):
        self.policy_old = PolicyNetwork(name='pi_old' , obs_dim=obs_dim , act_dim=act_dim , eta=eta , h_size=h_size)
        self.policy = PolicyNetwork(name='pi' , obs_dim=obs_dim , act_dim=act_dim , pi_old=self.policy_old , eta=eta ,
                                    h_size=h_size , kl_target=kl_target)
        self.value = ValueNetwork(name='vf' , obs_dim=obs_dim)
        self.kl_target = kl_target
        self.beta = beta
        self.lr_multiplier = 1.0
        # self.memory  =Memory(obs_dim, act_dim, max_steps)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_action_value(self , state):
        action , value = self.sess.run([self.policy.pi.sample() , self.value.vf] ,
                                       feed_dict={self.value.obs: [state] , self.policy.obs: [state]})
        return action.flatten() , value

    def get_pi(self , state):
        mu , logstd = self.sess.run([self.policy.mu , self.policy.logstd] , feed_dict={self.policy.obs: state})
        return mu , logstd

    def sync(self):
        self.sess.run(self.policy.sync)

    def train(self , dataset , num_iter=10 , lr=1e-3):
        # assert of class dataset
        # transfer current policy to old policy
        self.sync()
        for i in range(num_iter):
            for batch in dataset.iterate_once():
                feed_dict = {self.policy.obs: batch['obs'] , self.policy.adv: batch['adv'] ,
                             self.policy.acts: batch['acts'] ,
                             self.policy_old.obs: batch['obs'] , self.policy.beta: self.beta ,
                             self.policy.lr: self.lr_multiplier * lr}
                policy_loss , kl , e , _ = self.sess.run(
                    [self.policy.loss , self.policy.kl , self.policy.pi.entropy()[0] , self.policy.train] ,
                    feed_dict=feed_dict)
                if kl > 4 * self.kl_target:
                    tf.logging.info('KL too high. Stopping update after {}'.format(i))
                    break

        for i in range(num_iter):
            for batch in dataset.iterate_once():
                feed_dict = {self.value.obs: batch['obs'] , self.value.tdl: batch['tdl']}
                value_loss , _ = self.sess.run([self.value.loss , self.value.train] , feed_dict)

        self.update_beta(kl)

        return policy_loss , value_loss , kl , e

    # eps (0.7, 3/4)
    def update_beta(self , kl , alpha=1.5 , eps=(2. , 2.)):

        if kl > self.kl_target * eps[0]:
            # increasing penalty coefficient
            self.beta *= alpha
            tf.logging.info('KL too high, increasing penalty coefficient. Next Beta {}'.format(self.beta))
        elif kl < self.kl_target / eps[1]:
            # decrease penalty coeff
            self.beta /= alpha
            tf.logging.info('KL too low, decreasing penalty coefficient. Next Beta {}'.format(self.beta))
        else:
            tf.logging.info('We are close enough!')
            pass

    def update_beta_cody(self , kl , alpha=1.5 , eps=(2. , 2.)):
        """
        Like update_beta, with adaptive learning rate
        """
        if kl > self.kl_target * eps[0]:
            beta = np.minimum(35 , alpha * self.beta)

            if beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= alpha
            tf.logging.info('KL too high, increasing penalty coefficient. Next Beta {}'.format(self.beta))
        elif kl < self.kl_target / eps[1]:
            beta = np.maximum(1 / 35 , self.beta / alpha)
            if beta < (1 / 30) and self.lr_multiplier < 10:
                self.lr_multiplier *= alpha
            tf.logging.info('KL too low, decreasing penalty coefficient. Next Beta {}'.format(self.beta))
        else:
            tf.logging.info('We are close enough!')
            pass

    def close_session(self):
        self.sess.close()
