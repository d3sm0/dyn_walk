from value import ValueNetwork
from policy import PolicyNetwork
import tensorflow as tf
import numpy as np


class Agent(object):
    def __init__(self , obs_dim , act_dim , kl_target=0.003 , eta=50 , beta=1.0 , h_size=128):
        self.policy_old = PolicyNetwork(name= 'pi_old',obs_dim=obs_dim , act_dim=act_dim , eta=eta , h_size=h_size)
        self.policy = PolicyNetwork(name= 'pi',obs_dim=obs_dim , act_dim=act_dim , pi_old=self.policy_old , eta=eta ,
                                    h_size=h_size)
        self.value = ValueNetwork(name= 'vf',obs_dim=obs_dim)
        self.kl_target = kl_target
        self.beta = beta
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_action_value(self , state):
        action , value = self.sess.run([self.policy.pi.sample() , self.value.vf] ,
                                       feed_dict={self.value.obs: state , self.policy.obs: state})
        return action , value

    def get_pi(self , state):
        mu , logstd = self.sess.run([self.policy.mu , self.policy.logstd] , feed_dict={self.policy.obs: state})
        return mu , logstd

    def sync(self):
        self.sess.run(self.policy.sync)

    def train(self , dataset , num_ep=10):
        # assert of class dataset
        losses = []
        self.sync()
        for _ in range(num_ep):
            for batch in dataset.iterate_once():
                feed_dict = {self.policy.obs: batch['obs'] , self.policy.adv: batch['adv'] ,
                             self.policy_old: batch['obs'] , self.policy.beta: self.beta}
                policy_losses , kl , _ = self.sess.run([self.policy.losses , self.policy.kl , self.policy.train] ,
                                                       feed_dict=feed_dict)
                if kl.mean() > 4 * self.kl_target:
                    tf.logging.info('Policy update is going too far')
                    break

        for _ in range(num_ep):
            for batch in dataset.iterate_once():
                feed_dict = {self.value.obs: batch['obs'] , self.value.tdl: batch['tdl']}
                l , _ = self.sess.run([self.value.loss , self.value.train] , feed_dict)
                losses.append(l)

        self.update_beta(kl.mean())

    def update_beta(self , kl , alpha=1.5 , eps=(0.7 , 1.3)):

        if kl > self.kl_target * eps[0]:
            tf.logging.info('Increasing penalty coefficient')
            # increasing penalty coefficient
            self.beta *= alpha
        elif kl < self.kl_target * eps[1]:
            tf.logging.info('Decreasing penalty coefficient')
            # decrease penalty coeff
            self.beta /= alpha
        else:
            tf.logging.info('We are close enough!')
            pass

    # def update_beta_coady(self, kl,alpha = 1.5, eps = (2, 2)):
    #     """
    #     Like update_beta, with adaptive learning rate
    #     """
    #     if kl > self.kl_target * eps[0]:
    #         beta = np.minimum(35 , alpha * self.beta)
    #         if beta > 30 and self.lr_multiplier > 0.1:
    #             self.lr_multiplier /= alpha
    #     elif kl < self.kl_target / eps[1]:
    #         beta = np.maximum(1 / 35 , self.beta / alpha)
    #         if beta < (1 / 30) and self.lr_multiplier < 10:
    #             self.lr_multiplier *= alpha
    #     else:
    #         pass

    def close_session(self):
        self.sess.close()
