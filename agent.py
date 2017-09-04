import tensorflow as tf
import os
from ddpg import Actor, Critic, build_summaries, lrelu
import numpy as np
from utils.ou_noise import OUNoise
from memory.pmr import Experience
from enum import Enum
import multiprocessing
import gym
from utils.env_wrapper import EnvWrapper
import pickle


class SimulationCommands(Enum):
    step = 0
    observe = 1
    reset = 2


class IsolatedEnv(object):
    def __init__(self, env_name, experience_file, MAX_ACTIONS=100):
        self.steps = 0
        self.experience_file = open(experience_file, 'wb')
        self.env_name = env_name
        self.process, self.pipe = self.reset_simulator(env_name)
        self.MAX_ACTIONS = MAX_ACTIONS

    def reset_simulator(self, env_name):
        if self.pipe:
            self.pipe.close()
        if self.process:
            self.process.terminate()
        self.steps = 0
        self.experience_file.flush()

        pipe = multiprocessing.Pipe()
        pipe, pipe_process_end = pipe
        process = multiprocessing.Process(target=self.run_simulation, args=(env_name, pipe_process_end))
        process.start()
        return process, pipe

    @staticmethod
    def run_simulation(env_name, pipe):
        env = gym.make(env_name)
        while True:
            msg = pipe.recv()
            cmd = msg[0]
            args = msg[1:]
            if cmd == SimulationCommands.step:
                retr = env.step(args)
                pipe.send(retr)
            elif cmd == SimulationCommands.observe:
                pass
            elif cmd == SimulationCommands.reset:
                retr = env.reset()
                pipe.send(retr)

    def _avoid_memory_overflow(self):
        if self.steps > self.MAX_ACTIONS:
            self.process, self.pipe = self.reset_simulator(self.env_name)

    def __del__(self):
        self.process.terminate()
        self.experience_file.close()

    def reset(self):
        self.steps += 1
        self._avoid_memory_overflow()

        msg = (SimulationCommands.reset, None)
        self.pipe.send(msg)
        retr = self.pipe.recv()
        retr = retr.reshape(1, retr.shape[0])
        return retr

    def step(self, action):
        self.steps += 1
        msg = (SimulationCommands.step, action)
        self.pipe.send(msg)
        retr = self.pipe.recv()
        next_state, reward, terminal, _ = retr
        if self.experience_file:
            experience_tuple = (self.state, action, next_state)
            pickle.dump(self.experience_file, experience_tuple)
        if terminal:
            self._avoid_memory_overflow()

        retr = retr.reshape(1, retr.shape[0])
        return retr


class Agent(object):
    def __init__(self, name, log_dir, target=None, writer=None, split_obs=None, clip=20, summary_freq=5):

        # config
        self.summary_freq = summary_freq

        self.writer = writer
        self.gamma = 0.99
        self.clip = clip

        self.name = name
        self.target = target
        self.log_dir = log_dir

        tf.logging.info('Worker {} ready to go ...'.format(self.name))

    def initialize(self, env_dims, writer=None, h_size=128, batch_size=32, memory_size=10e6,
                   policy='det', act=tf.nn.elu, split_obs=None):

        self.obs_space, self.act_space, self.bound = env_dims

        self.writer = tf.summary.FileWriter(self.log_dir)

        self.memory = Experience(batch_size=batch_size, memory_size=int(memory_size))

        with tf.variable_scope(self.name):
            self.actor = Actor(self.obs_space, self.act_space, self.bound, h_size=h_size, policy=policy, act=act,
                               split_obs=split_obs)
            self.critic = Critic(self.obs_space, self.act_space, h_size=h_size, act=act)

        self.sess = tf.Session()

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name), max_to_keep=2)

        self.sync_op = [self.update_target(self.actor.params, self.target.actor.params),
                        self.update_target(self.critic.params, self.target.critic.params)]

        scalar = [self.critic.q, self.critic.critic_loss]
        hist = [self.critic.state, self.critic.action, self.actor.grads]

        self.summary_ops = build_summaries(scalar=scalar, hist=hist)
        self.ou = OUNoise(action_dimension=self.act_space)

        self.sess.run(tf.global_variables_initializer())
        # copy params for
        self.copy_params_from_target(self.target)

    def add_memory(self, tuple, priority):
        self.memory.add(tuple, priority)

    def __del__(self):
        self.sess.close()

    def save_progress(self):
        global_step = self.sess.run(tf.contrib.framework.get_global_step())
        self.saver.save(self.sess, os.path.join(self.log_dir, 'model.ckpt'), global_step=global_step)
        tf.logging.info('Model saved at ep {}'.format(global_step))

    def summarize(self, feed_dict, global_step):

        summary = self.sess.run(self.summary_ops,
                                feed_dict=feed_dict)

        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def update_target_network(self):

        self.sess.run(self.sync_op)

    def train(self, state, action, q):
        critic_loss, _, global_step = self.sess.run(
            [self.critic.critic_loss, self.critic.train, tf.contrib.framework.get_global_step()], feed_dict={
                self.critic.state: state,
                self.critic.action: action,
                self.critic.q: q
            })

        # compute sample of the gradient
        sampled_action = self.get_action(state)
        sampled_grads = self.get_grads(state, sampled_action)

        _, = self.sess.run([self.actor.train], feed_dict={
            self.actor.state: state,
            self.actor.grads: sampled_grads
        })

        if global_step % self.summary_freq == 0:
            feed_dict = {
                self.actor.state: state,
                self.actor.grads: sampled_grads,
                self.critic.state: state,
                self.critic.action: action,
                self.critic.q: q
            }
            self.summarize(feed_dict, global_step)

        return critic_loss

    def get_grads(self, state, action):
        grads, = self.sess.run(self.critic.action_grads, feed_dict={
            self.critic.state: state,
            self.critic.action: action,
        })
        return grads

    def get_action(self, state):
        # TODO pre-process of state should not happen here
        if np.ndim(state) != self.obs_space:
            state = np.reshape(state, (-1, self.obs_space))

        mu_hat = self.sess.run(self.actor.mu_hat,
                               feed_dict={self.actor.state: state})
        return mu_hat

    def get_q(self, state, action):

        if np.ndim(state) != self.obs_space or np.ndim(action) != self.act_space:
            state = np.reshape(state, (-1, self.obs_space))
            action = np.reshape(action, (-1, self.act_space))

        q_hat = self.sess.run(self.critic.q_hat, feed_dict={self.critic.state: state, self.critic.action: action})

        if self.clip:
            q_hat = np.clip(q_hat, -self.clip, self.clip)

        return q_hat.ravel()

    def get_td(self, state, action, reward, next_state, terminal):
        # TODO: IPC
        target_action = self.target.get_action(next_state)
        target_q = self.target.get_q(next_state, target_action)[0]

        local_q = self.get_q(state, action)[0]
        q = reward

        if not terminal:
            q += self.gamma * target_q
        return np.abs(q - local_q), q

    def think(self):
        data, _, idxs = self.memory.select()
        if data is not None:
            s1_batch, a_batch, r_batch, s2_batch, t_batch = data
            q = self.get_q_batch(a_batch, idxs, r_batch, s1_batch, s2_batch, t_batch)
            self.train(s1_batch, a_batch, q)

    def get_q_batch(self, a_batch, idxs, r_batch, s1_batch, s2_batch, t_batch):
        target_action = self.target.get_action(s2_batch)
        target_q = self.target.get_q(s2_batch, target_action)
        local_q = self.get_q(s1_batch, a_batch)
        # TODO check data structure here
        q = []
        td = []
        for k in range(self.memory.batch_size):
            if t_batch[k]:
                q.append(r_batch[k])
                td.append(r_batch[k])
            else:
                q.append(r_batch[k] + self.gamma * target_q[k])
                td.append(r_batch[k] + self.gamma * target_q[k] - local_q[k])
        self.memory.priority_update(indices=idxs, priorities=np.abs(td))
        q = np.reshape(q, (self.memory.batch_size, 1))
        return q

    @staticmethod
    # tau = 0.001
    def update_target(local, target, tau=0.01):
        params = []
        for i in range(len(target)):
            params.append(
                target[i].assign(tf.multiply(local[i], tau) + tf.multiply(target[i], 1. - tau)))
        return params

    @staticmethod
    def global_trainer(grads, params):
        grads = [grad for grad in grads if grad is not None]
        grads, _ = tf.clip_by_global_norm(grads, 40)
        train_op = tf.train.AdamOptimizer().apply_gradients(zip(grads, params),
                                                            global_step=tf.contrib.framework.get_global_step())
        return train_op

    def copy_params_from_target(self, target_network=None):
        assert target_network is not None
        self.update_target(self.critic.params, target_network.critic.params, tau=1)
        self.update_target(self.actor.params, target_network.actor.params, tau=1)

    def init(self):
        self.sess.run(tf.global_variables_initializer())

    def reset(self):
        self.ou.reset()

    def process(self, state, action, reward, next_state, terminal):
        if self.clip:
            reward = np.clip(reward, -self.clip, self.clip)

        td, q = self.get_td(state, action, reward, next_state, terminal)
        self.add_memory((state, action, reward, next_state, terminal), priority=td)
        self.think()
        return td, q


class Target(Agent):
    def __init__(self, log_dir, split_obs=None):
        # config
        super(Target, self).__init__(
            name="target",
            log_dir=log_dir,
            target=None,
            writer=None,
            split_obs=split_obs,
            clip=None,
            summary_freq=None
        )

    def copy_params_from_target(self, target_network=None):
        raise NotImplementedError()

    def initialize(self, env_dims, writer=None, h_size=128, batch_size=32, memory_size=10e6,
                   policy='det', act=tf.nn.elu, split_obs=None):

        self.obs_space, self.act_space, self.bound = env_dims

        with tf.variable_scope(self.name):
            self.actor = Actor(self.obs_space, self.act_space, self.bound, h_size=h_size, policy=policy, act=act,
                               split_obs=split_obs)
            self.critic = Critic(self.obs_space, self.act_space, h_size=h_size, act=act)

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name), max_to_keep=2)
        self.load_from_ckpt()
        tf.logging.info('Target {} ready to go ...'.format(self.name))

    def add_memory(self, tuple, priority):
        raise NotImplementedError()

    def __del__(self):
        self.sess.close()

    def save_progress(self):
        # global_step = self.sess.run(tf.contrib.framework.get_global_step())
        # self.saver.save(self.sess, os.path.join(self.log_dir, 'model.ckpt'), global_step=global_step)
        # tf.logging.info('Model saved at ep {}'.format(global_step))
        raise NotImplementedError()

    def summarize(self, feed_dict, global_step):

        # summary = self.sess.run(self.summary_ops,
        #                    feed_dict=feed_dict)
        # self.writer.add_summary(summary, global_step=global_step)
        # self.writer.flush()
        raise NotImplementedError()

    def update_target_network(self):
        raise NotImplementedError()

    def train(self, state, action, q):
        raise NotImplementedError()

    def get_grads(self, state, action):
        raise NotImplementedError()

    def get_action(self, state):
        # TODO pre-process of state should not happen here
        if np.ndim(state) != self.obs_space:
            state = np.reshape(state, (-1, self.obs_space))

        mu_hat = self.sess.run(self.actor.mu_hat,
                               feed_dict={self.actor.state: state})
        return mu_hat

    def get_q(self, state, action):

        if np.ndim(state) != self.obs_space or np.ndim(action) != self.act_space:
            state = np.reshape(state, (-1, self.obs_space))
            action = np.reshape(action, (-1, self.act_space))

        q_hat = self.sess.run(self.critic.q_hat, feed_dict={self.critic.state: state, self.critic.action: action})

        if self.clip:
            q_hat = np.clip(q_hat, -self.clip, self.clip)

        return q_hat.ravel()

    def get_td(self, state, action, reward, next_state, terminal):
        raise NotImplementedError()

    def think(self):
        raise NotImplementedError()

    def get_q_batch(self, a_batch, idxs, r_batch, s1_batch, s2_batch, t_batch):
        raise NotImplementedError()

    @staticmethod
    # tau = 0.001
    def update_target(local, target, tau=0.01):
        raise NotImplementedError()

    @staticmethod
    def global_trainer(grads, params):
        raise NotImplementedError()

    def load_from_ckpt(self):

        ckpt = tf.train.latest_checkpoint(self.log_dir)

        if ckpt is not None:
            try:
                tf.logging.info('Restore model {}'.format(ckpt))
                self.saver.restore(sess=self.sess, save_path=ckpt)
            except Exception as e:
                tf.logging.info("Failed restore {}".format(e))
                raise e

    def reset(self):
        pass

    def process(self, state, action, reward, next_state, terminal):
        raise NotImplementedError()


class Worker(object):
    def __init__(self, target, config, log_dir):
        self.env, env_dim, _ = self.init_environment(config)
        self.agent = Agent(name='local', target=target, split_obs=None, log_dir=log_dir)

        self.agent.initialize(env_dims=env_dim, h_size=config['H_SIZE'],
                              policy=config['POLICY'], act=eval(config['ACTIVATION']), split_obs=None)
        self.writer = tf.summary.FileWriter(self.agent.log_dir, filename_suffix="episode_metrics")
        self.ep_summary = tf.Summary()

    def unroll(self):
        state = self.env.reset()
        self.agent.reset()

        terminal = False
        timesteps, tot_rw, tot_q, tot_td = 0, 0, 0, 0
        while not terminal:
            # if stochastic remove exploration noise
            action = self.agent.get_action(state).flatten() + self.agent.ou.noise()
            next_state, reward, terminal, _ = self.env.step(action)
            THE_TUPLE = (state, action, reward, next_state, terminal)

            td, q = self.agent.process(*THE_TUPLE)

            state = next_state

            timesteps += 1
            tot_td += td
            tot_q += q
            tot_rw += reward
        return tot_td, tot_q, tot_rw, timesteps

    def report_metrics(self, tot_td, tot_q, tot_rw, timesteps, episode_count):
        self.ep_summary.value.add(simple_value=tot_rw / timesteps, tag='eval/avg_surr_rw')
        self.ep_summary.value.add(simple_value=tot_q / timesteps, tag='eval/avg_q')
        self.ep_summary.value.add(simple_value=tot_td / timesteps, tag='eval/avg_td')
        self.ep_summary.value.add(simple_value=tot_rw, tag='eval/surr_rw')
        self.ep_summary.value.add(simple_value=tot_td, tag='eval/total_td')
        self.ep_summary.value.add(simple_value=tot_q, tag='eval/total_q')
        self.ep_summary.value.add(simple_value=timesteps, tag='eval/ep_length')
        try:
            self.ep_summary.value.add(simple_value=self.env.r / timesteps, tag='eval/avg_rw')
            self.ep_summary.value.add(simple_value=self.env.r, tag='eval/total_rw')
        except AttributeError:
            pass
        self.writer.add_summary(self.ep_summary, episode_count)
        self.writer.flush()

    @staticmethod
    def init_environment(config):
        split_obs = None
        if config['ENV_NAME'] == 'osim':
            try:
                env = EnvWrapper(RunEnv, visualize=False, augment_rw=config['USE_RW'],
                                 concat=config['CONCATENATE_FRAMES'], add_time=False,
                                 normalize=config['NORMALIZE'], add_acceleration=7)
                env_dims = env.get_dims()
                split_obs = env.split_obs
            except Exception as e:
                tf.logging.info('Environment not found')
                raise e
        else:
            try:
                import gym
                env = gym.make(config['ENV_NAME'])
                env_dims = (
                    env.observation_space.shape[0], env.action_space.shape[0],
                    (env.action_space.low, env.action_space.high))
            except:
                raise NotImplementedError()
        return env, env_dims, split_obs
