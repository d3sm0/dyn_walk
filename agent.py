import numpy as np
import tensorflow as tf
import os

from policy import PolicyNetwork
from utils.misc_utils import explained_variance
from value import ValueNetwork


class Agent(object):
    def __init__(self , obs_dim , act_dim , kl_target=1e-1 , eta=1000 , beta=1.0 , h_size=(128 , 64 , 32)):
        self.policy = PolicyNetwork(name='pi' , obs_dim=obs_dim , act_dim=act_dim , eta=eta ,
                                    h_size=h_size , kl_target=kl_target)
        self.value = ValueNetwork(name='vf' , obs_dim=obs_dim , h_size=h_size)
        self.kl_target = kl_target
        self.beta = beta
        self.lr_multiplier = 1.0
        self.early_stop = 0
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=self.policy.get_params() + self.value.get_params(), max_to_keep=2)

    def save(self, log_dir):
        try:
            self.saver.save(sess=self.sess, save_path=os.path.join(log_dir, 'model.ckpt'))
        except Exception as e:
            tf.logging.error(e)
            raise
    def load(self, log_dir):
        try:
            ckpt = tf.train.latest_checkpoint(log_dir)
            self.saver.restore(sess= self.sess, save_path=ckpt)
        except Exception as e:
            tf.logging.error(e)
            raise

    def get_action(self , state):
        return self.sess.run(self.policy.sample , feed_dict={self.policy.obs: [state]}).flatten()

    def get_value(self , state):
        return self.sess.run(self.value.vf , feed_dict={self.value.obs: state})

    def get_action_value(self , state):
        action , value = self.sess.run([self.policy.sample , self.value.vf] ,
                                       feed_dict={self.value.obs: [state] , self.policy.obs: [state]})
        return action.flatten() , value

    def get_pi(self , state):
        mu , logstd = self.sess.run([self.policy.mu , self.policy.logstd] , feed_dict={self.policy.obs: state})
        return mu , logstd

    def train(self , dataset ,num_iter=10 , lr=1e-3, eps= (2.,2.)):

        dataset.data['mu_old'] , logstd_old = self.get_pi(dataset.data['obs'])

        for i in range(num_iter):

            for batch in dataset.iterate_once():
                feed_dict = {self.policy.obs: batch['obs'] , self.policy.adv: batch['adv'] ,
                             self.policy.acts: batch['acts'] , self.policy.mu_old: batch['mu_old'] ,
                             self.policy.logstd_old: logstd_old ,
                             self.policy.beta: self.beta ,
                             self.policy.lr: self.lr_multiplier * lr}

                policy_loss , kl , e , _ = self.sess.run(
                    [self.policy.loss , self.policy.kl , self.policy.entropy , self.policy.train] ,
                    feed_dict=feed_dict)

            if kl > 4 * self.kl_target:
                tf.logging.info('KL too high. Stopping update after {}'.format(i))
                self.early_stop += 1
                break

        for i in range(num_iter):
            for batch in dataset.iterate_once():
                feed_dict = {self.value.obs: batch['obs'] , self.value.tdl: batch['tdl'] ,
                             self.value.value_old: batch['vs']}
                value_loss , _ = self.sess.run([self.value.loss , self.value.train] , feed_dict)

        self.update_beta(kl, eps=eps)

        stats = {
            '_pl': policy_loss ,
            '_vl': value_loss ,
            '_kl': kl ,
            '_e': e ,
            '_expl_var': explained_variance(dataset.data['vs'] , dataset.data['tdl']) ,
            '_beta': self.beta ,
            '_early_stop': self.early_stop
        }

        return stats

    # eps (0.7, 3/4)
    def update_beta(self , kl , alpha=1.5 , eps=(2. , 2.)):

        if kl > self.kl_target * eps[0]:
            # increasing penalty coefficient
            self.beta *= alpha
            # tf.logging.info('KL too high, increasing penalty coefficient. Next Beta {}'.format(self.beta))
        elif kl < self.kl_target / eps[1]:
            # decrease penalty coeff
            self.beta /= alpha
            # tf.logging.info('KL too low, decreasing penalty coefficient. Next Beta {}'.format(self.beta))
        else:
            # tf.logging.info('We are close enough!')
            pass

    def update_beta_cody(self , kl , alpha=1.5 , eps=(2. , 2.)):
        """
        Like update_beta, with adaptive learning rate
        """
        if kl > self.kl_target * eps[0]:
            self.beta = np.minimum(35 , alpha * self.beta)

            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= alpha
            tf.logging.info('KL too high, increasing penalty coefficient. Next Beta {}, lr {}'.format(self.beta ,
                                                                                                      self.lr_multiplier))
        elif kl < self.kl_target / eps[1]:
            self.beta = np.maximum(1 / 35 , self.beta / alpha)
            if self.beta < (1 / 30) and self.lr_multiplier < 10:
                self.lr_multiplier *= alpha
            tf.logging.info('KL too low, decreasing penalty coefficient. Next Beta {}, lr {}'.format(self.beta ,
                                                                                                     self.lr_multiplier))
        else:
            tf.logging.info('We are close enough!')
            pass

    def close_session(self):
        self.sess.close()

