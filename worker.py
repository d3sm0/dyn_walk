from utils.env_wrapper import Environment
import tensorflow as tf
from collections import deque
import os
from agent import Agent
import numpy as np
from memory.dataset import Memory


class Worker(object):
    def __init__(self , config , log_dir):
        self.env , self.env_dim , split_obs = self.init_environment(config)

        self.agent = Agent(obs_dim=self.env_dim[0] , act_dim=self.env_dim[1] , kl_target=config['KL_TARGET'] ,
                           eta=config['ETA'] ,
                           beta=config['BETA'] , h_size=config['H_SIZE'])
        self.gamma = config['GAMMA']
        self.lam = config['LAMBDA']
        self.memory = Memory(obs_dim=self.env_dim[0] , act_dim=self.env_dim[1] , max_steps=config['MAX_STEPS_BATCH'] ,
                             main_path=log_dir)
        self.writer = tf.summary.FileWriter(logdir=log_dir)
        self.ep_summary = tf.Summary()

        if config['LAST_RUN'] != False:
            load_path = os.path.join(os.getcwd(), 'log-files', config['ENV_NAME'], config['LAST_RUN'])
            self.agent.load(load_path)

    def warmup(self , ob_filter=None , max_steps=64 , ep=1):
        for e in range(ep):
            ob = self.env.reset()
            for _ in range(max_steps):
                if ob_filter is not None: ob = ob_filter(ob)
                a = self.agent.get_action(ob)
                ob , r , t , _ = self.env.step(a)
                if t:
                    ob = self.env.reset()

        tf.logging.info('Warmup ended')

    def eval(self , log_dir):

        ob = self.env.reset()
        vs , rs , i = 0 , 0 , 0
        self.agent.load(log_dir)
        t = False
        while not t:
            self.env.render()
            a , v = self.agent.get_action_value(ob)
            ob , r , t , _ = self.env.step(a)
            vs += v
            rs += r
            i += 1

        return {'rs': rs , 'vs': vs , 'i': i}

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
        batch = seq.copy()
        # TODO can i compute the advatnage like this or should i use a variable and then reassign it?
        # standardize advantage. Should not change the element in seq
        batch['adv'] = (batch['adv'] - batch['adv'].mean()) / batch['adv'].std()
        return batch

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
        # TODO save only last 10 episodes using deque
        ep_rws , ep_ls = deque(maxlen = 10), deque(maxlen = 10)
        while True:

            if ob_filter: ob = ob_filter(ob)

            act , v = self.agent.get_action_value(ob)

            if t > 0 and t % max_steps == 0:
                yield self.memory.release(v=v , done=done , t=t) , self.compute_summary(ep_l,ep_r,ep_rws , ep_ls , ep , t)
                ep_rws = []
                ep_ls = []
            ob1 , r , done , _ = self.env.step(act)
            # TODO do i need to deep copy here?
            self.memory.collect((ob.copy() , act , r , done , v) , t)
            ob = ob1

            ep_l += 1
            ep_r += r
            if done:
                ep += 1
                ep_rws.append(ep_r)
                ep_ls.append(ep_l)
                ep_r = 0
                ep_l = 0
                ob = self.env.reset()
            t += 1
    @staticmethod
    def compute_summary(*stats):
        ep_l,ep_r, ep_rws , ep_ls , ep , t = stats

        ep_stats = {
            'last_ep_rw': ep_r,
            'last_ep_len': ep_l,
            'avg_rw': np.array(ep_rws).mean() ,
            'avg_len': np.array(ep_ls).mean() ,
            'total_steps': t ,
            'total_ep': ep ,
        }
        return ep_stats

    def write_summary(self , stats , ep , network_stats=None):

        for (k , v) in stats.iteritems():
            # if np.ndim(v) > 1:
            #     self.ep_summary.value(simple_value=v.mean() , tag='batch_{}_mean'.format(k))
            #     self.ep_summary.value(simple_value=v.max() , tag='batch_{}_max'.format(k))
            #     self.ep_summary.value(simple_value=v.min() , tag='batch_{}_min'.format(k))
            # else:
                self.ep_summary.value.add(simple_value=v , tag=k)
        if network_stats is not None:
            self.writer.add_summary(network_stats , ep)
        self.writer.add_summary(self.ep_summary , ep)
        self.writer.flush()
