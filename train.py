from datetime import datetime

import gym
import numpy as np
import tensorflow as tf

from agent import Agent
from memory.dataset import Dataset , Memory
from utils.env_wrapper import Environment
from utils.logger import Logger
from utils.running_stats import ZFilter

tf.logging.set_verbosity(tf.logging.INFO)


# remember with no trustd region you tested 128 256 512, wuth trusted region you tested 512 256 128
def main():
    env_name = 'Walker2d-v1'
    env , env_dims = create_env(env_name)
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    logger = Logger(logname=env_name , now=now)
    agent = Agent(obs_dim=env_dims[0] , act_dim=env_dims[1] , kl_target=1e-2 , log_dir=logger.path)
    play(agent=agent , env=env , env_dims=env_dims , max_ep=10000 , seq_len=2048 , logger=logger)


def warmup(env , ob_filter , max_steps=256 , ep=1):
    for e in range(ep):
        ob = env.reset()
        for _ in range(max_steps):
            a = env.action_space.sample()
            ob = ob_filter(ob)
            ob , r , t , _ = env.step(a)
            if t:
                ob = env.reset()

    tf.logging.info('Warmup ended')


def create_env(env_name):
    if env_name == 'osim':
        env = Environment(normalize=False)
        env_dims = env.get_dims()
    else:
        env = gym.make(env_name)
        env_dims = (
            env.observation_space.shape[0] , env.action_space.shape[0] , (env.action_space.low , env.action_space.high))
    return env , env_dims


def play(agent , env , env_dims , max_ep , seq_len=256 , logger=None):
    ob_filter = ZFilter((env_dims[0] ,))
    memory = Memory(env_dims[0] , env_dims[1] , max_steps=seq_len)
    seq_gen = unroll(env , agent , memory , ob_filter=ob_filter , max_steps=seq_len , logger=logger)
    warmup(env , ob_filter , max_steps=seq_len)
    saver = tf.train.Saver(agent.policy.get_params() + agent.value.get_params() , max_to_keep=2)
    tf.logging.info('Init training')

    steps = 0
    last_save = 0
    ep = 0
    while ep < max_ep:
        sequence , t , ep = next(seq_gen)
        steps = t
        compute_target(sequence)

        b = Dataset(dict(obs=sequence['obs'] , acts=sequence['acts'] , adv=sequence['adv'] , tdl=sequence['tdl'] ,
                         vs=sequence['vs']) ,
                    shuffle=True)

        stats = agent.train(b , num_iter=20)
        logger.log(stats)
        if steps % 2048 != 0:
            logger.write(display=False)
        else:
            logger.write(display=True)
        if ep - last_save > 400:
            saver.save(sess=agent.sess , global_step=ep , save_path=logger.path + '/model.ckpt')
            tf.logging.info('Saving model at ep {}'.format(ep))
            last_save = ep


def unroll(env , agent , memory , max_steps=2048 , ob_filter=None , logger=None):
    ob = env.reset()
    t , ep , ep_r , l = 0 , 0 , 0 , 0

    while True:

        if ob_filter: ob = ob_filter(ob)

        act , v = agent.get_action_value(ob)

        if t > 0 and t % max_steps == 0:
            yield memory.release(v=v , done=done) , t , ep

        ob1 , r , done , _ = env.step(act)
        memory.collect((ob , act , r , done , v) , t)
        ob = ob1.copy()
        t += 1
        l += 1
        ep_r += r
        if done:
            ep += 1
            logger.log({'_ep_len': l , '_ep_rw': ep_r , '_ep': ep , '_steps': t})
            ob = env.reset()
            ep_r = 0
            l = 0


def compute_target(seq , gamma=0.99 , lam=0.95):
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
        delta = rws[t] + gamma * v_hat[t + 1] * not_terminal - v_hat[t]
        gae[t] = last_gae = delta + gamma * lam * not_terminal * last_gae
    seq['tdl'] = seq['adv'] + seq['vs']
    # seq['tdl'] = discount(rws , gamma)
    # standardized advantage function
    seq['adv'] = (seq['adv'] - seq['adv'].mean()) / seq['adv'].std()


if __name__ == '__main__':
    main()
