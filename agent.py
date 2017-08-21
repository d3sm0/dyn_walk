import tensorflow as tf
from ddpg import Actor, Critic, build_summaries
import numpy as np
from replay_buffer import ReplayBuffer


# reverse the paradigm: experience with local agents, compute gradients locally, but apply to the global network
# after T iteration, copy the global params down to the local network..

# def train_op(global_vars , gradients , optim=tf.train.RMSPropOptimizer ( 0.0001 ) , max_clip=40.0):
#     grads_clipped , _ = tf.clip_by_global_norm ( gradients , max_clip )
#     return optim.apply_gradients ( list ( zip ( grads_clipped , global_vars ) ) ,
#                                    global_step= )


class Agent(object):
    def __init__(self, name, env_dim, target_network_lock, target=None, writer=None, h_size=128, memory_size=10e6,
                 stochastic=False,):
        self.target_network_lock = target_network_lock

        obs_space, act_space, bound = env_dim
        with tf.variable_scope(name):
            # TODO: unify networks
            self.actor = Actor(obs_space, act_space, bound, h_size=h_size, stochastic=stochastic)
            self.critic = Critic(obs_space, act_space, h_size=h_size)

        # TODO Use share memory between worker
        # TODO test prioritized experience replay
        self.memory = ReplayBuffer(env_shape=env_dim, buffer_size=memory_size)

        if name != 'target':
            # reverse this, from global to local
            # self.sync_op = [ self.update_target ( self.actor.params , target.actor.params ) ,
            #                  self.update_target ( self.critic.params , target.critic.params ) ]

            self.sync_op = [self.update_target(target.actor.params, self.actor.actor_params),
                            self.update_target(target.critic.params, self.critic.params)]

            self.train_target = self.global_trainer((self.actor.actor_grads + self.critic.critic_grads),
                                                    target.critic.params)

            self.summary_ops = build_summaries(scalar=[self.critic.q, self.critic.critic_loss],
                                               hist=[self.critic.state, self.critic.action,
                                                     self.actor.action_grads])

            self.writer = writer
            self.target = target

        self.name = name
        self.obs_space = env_dim[0]
        self.act_space = env_dim[1]
        self.h_size = h_size

        tf.logging.info('Worker {} ready to go ...'.format(self.name))

    def summarize(self, feed_dict, global_step):

        sess = tf.get_default_session()

        summary = sess.run(self.summary_ops,
                           feed_dict=feed_dict)

        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def sync(self):
        # tf.logging.info('Updating target network')
        sess = tf.get_default_session()
        # TODO: thread safe
        with self.target_network_lock as _:
            sess.run(self.sync_op)

    def train(self, state, action, q, get_summary=None):

        sess = tf.get_default_session()

        global_step, critic_loss = sess.run(
            [tf.contrib.framework.get_global_step(), self.critic.critic_loss],
            feed_dict={
                self.critic.state: state,
                self.critic.action: action,
                self.critic.q: q
            })

        sampled_action = self.get_action(state, )
        sampled_grads = self.get_grads(state, sampled_action)

        _ = sess.run(self.train_target, feed_dict={self.actor.state: state, self.actor.action_grads: sampled_grads,
                                                   self.critic.state: state,
                                                   self.critic.action: action, self.critic.q: q})

        if get_summary:
            feed_dict = {
                self.actor.state: state,
                self.actor.action_grads: sampled_grads,
                self.critic.state: state,
                self.critic.action: action,
                self.critic.q: q
            }
            self.summarize(feed_dict, global_step)

        return critic_loss

    #
    # def train(self , state , action , q , get_summary=False):
    #
    #     sess = tf.get_default_session ()
    #
    #     critic_loss , _ , global_step = sess.run (
    #         [ self.critic.critic_loss , self.critic.train , tf.contrib.framework.get_global_step () ] , feed_dict={
    #             self.critic.state: state ,
    #             self.critic.action: action ,
    #             self.critic.q: q
    #         } )
    #
    #     # compute sample of the gradient
    #
    #     sampled_action = self.get_action ( state )
    #     sampled_grads = self.get_grads ( state , sampled_action )
    #
    #     _ , = sess.run ( [ self.actor.train ] , feed_dict={
    #         self.actor.state: state ,
    #         self.actor.grads: sampled_grads
    #     } )
    #
    #     if get_summary:
    #         feed_dict = {
    #             self.actor.state: state ,
    #             self.actor.grads: sampled_grads ,
    #             self.critic.state: state ,
    #             self.critic.action: action ,
    #             self.critic.q: q
    #         }
    #         self.summarize ( feed_dict , global_step )
    #
    #     return critic_loss

    def get_grads(self, state, action):
        sess = tf.get_default_session()
        return sess.run(self.critic.action_grads, feed_dict={
            self.critic.state: state,
            self.critic.action: action,
        })[0]

    def get_action(self, state):
        sess = tf.get_default_session()

        # TODO pre-process of state should not happen here
        if np.ndim(state) != self.obs_space:
            state = np.reshape(state, (-1, self.obs_space))

        mu_hat = sess.run(self.actor.mu_hat,
                          feed_dict={self.actor.state: state})

        return mu_hat

    def get_q(self, state, action):

        sess = tf.get_default_session()
        if np.ndim(state) != self.obs_space:
            state = np.reshape(state, (-1, self.obs_space))

        q_hat = sess.run(self.critic.q_hat, feed_dict={self.critic.state: state, self.critic.action: action})
        return q_hat

    def think(self, gamma, batch_size, summarize):

        if self.memory.get_size() > batch_size:
            s1_batch, a_batch, r_batch, s2_batch, t_batch = self.memory.get_sample(batch_size)

            target_action = self.target.get_action(s2_batch)
            target_q = self.target.get_q(s2_batch, target_action)

            y_i = []

            for k in range(batch_size):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + gamma * target_q[k])

            y_i = np.reshape(y_i, (batch_size, 1))

            c_loss = self.train(s1_batch, a_batch, y_i, get_summary=summarize)

    @staticmethod
    # tau = 0.001
    def update_target(local, target, tau=1):
        params = []
        for l, t in zip(local, target):
            params.append(t.assign(tf.multiply(l, tau) + tf.multiply(t, 1. - tau)))

        return params

    @staticmethod
    def global_trainer(grads, params):
        grads = [grad for grad in grads if grad is not None]
        grads, _ = tf.clip_by_global_norm(grads, 40)
        train_op = tf.train.AdamOptimizer().apply_gradients(zip(grads, params),
                                                            global_step=tf.contrib.framework.get_global_step())
        return train_op
