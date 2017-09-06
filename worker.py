from utils.env_wrapper import Environment
import tensorflow as tf
from agent import Agent
import numpy as np
from memory.dataset import Memory
from collections import deque


class Worker(object):
    def __init__(self , config , log_dir):
        self.env , self.env_dim , split_obs = self.init_environment(config)

        self.agent = Agent(obs_dim=self.env_dim[0], act_dim=self.env_dim[1], kl_target=config['KL_TARGET'] , eta=config['ETA'] ,
                           beta=config['BETA'] , h_size=config['H_SIZE'])
        self.gamma = config['GAMMA']
        self.lam = config['LAMBDA']
        self.memory = Memory(obs_dim=self.env_dim[0], act_dim=self.env_dim[1] , max_steps=config['MAX_STEPS'])
        self.writer = tf.summary.FileWriter(logdir=log_dir)

    def warmup(self , ob_filter=None , max_steps=256 , ep=1):
        for e in range(ep):
            ob = self.env.reset()
            for _ in range(max_steps):
                if ob_filter is not None: ob = ob_filter(ob)
                a = self.agent.get_action(ob)
                ob , r , t , _ = self.env.step(a)
                if t:
                    ob = self.env.reset()

        tf.logging.info('Warmup ended')

    def compute_target(self , seq):
        # TODO not sure why using the tdl estimator instead of the discounted sum of rewards
        dones = np.append(seq['ds'] , 0)
        v_hat = np.append(seq['vs'] , seq['v_next'])
        T = len(seq['rws'])
        seq['adv'] = gae = np.empty(T , 'float32')
        rws = seq['rws']
        last_gae = 0
        for t in reversed(range(T)):
            not_terminal = 1 - dones[t + 1]
            # td error
            delta = rws[t] + self.gamma * v_hat[t + 1] * not_terminal - v_hat[t]
            gae[t] = last_gae = delta + self.gamma * self.lam * not_terminal * last_gae
        seq['tdl'] = seq['adv'] + seq['vs']
        # seq['tdl'] = discount(rws , self.gamma)
        # standardized advantage function
        seq['adv'] = (seq['adv'] - seq['adv'].mean()) / seq['adv'].std()

    @staticmethod
    def init_environment(config):
        env = None
        env_dims = None
        split_obs = None
        if config['ENV_NAME'] == 'osim':
            try:
                env = Environment(augment_rw=config['USE_RW'] ,
                                  concat=config['CONCATENATE_FRAMES'] ,
                                  normalize=config['NORMALIZE'] ,
                                  frame_rate=config['FRAME_RATE'])
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
                    env.observation_space.shape[0] , env.action_space.shape[0] ,
                    (env.action_space.low , env.action_space.high))
            except:
                raise NotImplementedError()
        return env , env_dims , split_obs

    def unroll(self , max_steps=2048 , ob_filter=None):
        ob = self.env.reset()
        t , ep , ep_r , ep_l = 0 , 0 , 0 , 0
        ep_rws , ep_ls = deque(maxlen=10) , deque(maxlen=10)
        while True:

            if ob_filter: ob = ob_filter(ob)

            act , v = self.agent.get_action_value(ob)

            if t > 0 and t % max_steps == 0:
                yield self.memory.release(v=v , done=done) , self.compute_summary(ep_rws , ep_ls , ep , t)
                ep_rws.clear()
                ep_ls.clear()
            ob1 , r , done , _ = self.env.step(act)
            self.memory.collect((ob , act , r , done , v) , t)
            ob = ob1.copy()

            ep_l += 1
            ep_r += r
            if done:
                ep += 1
                ob = self.env.reset()
                ep_rws.append(ep_r)
                ep_ls.append(ep_l)
                ep_r =0
                ep_l = 0
            t += 1

    def compute_summary(self , *stats):
        ep_rws , ep_ls , ep , t = stats

        ep_stats = {
            'rw': ep_rws[-1] ,
            'len': ep_ls[-1] ,
            'avg_rw': np.array(ep_rws).mean() ,
            'avg_len': np.array(ep_ls).mean() ,
            't': t ,
            'ep': ep
        }
        return ep_stats


    def write_summary(self, stats, ep):
        ep_summary = tf.Summary()
        for (k,v) in stats.iteritems():
            ep_summary.value.add(simple_value = v, tag = k)

        self.writer.add_summary(ep_summary , ep)
        self.writer.flush()

