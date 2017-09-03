import gym
from dataset import Dataset , Memory
import numpy as np
from agent import Agent
import tensorflow as tf
from misc_utils import ZFilter , explained_variance , discount

tf.logging.set_verbosity(tf.logging.INFO)


def main():
    env_name = 'Walker2d-v1'
    env , env_dims = create_env(env_name)
    agent = Agent(obs_dim=env_dims[0] , act_dim=env_dims[1] , kl_target=1e-2)
    play(agent=agent , env=env , env_dims=env_dims , max_steps=128)


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


def play(agent , env , env_dims , max_ep=1000 , max_steps=256):
    ob_filter = ZFilter((env_dims[0] ,))
    memory = Memory(env_dims[0] , env_dims[1] , max_steps =max_steps)
    seq_gen = unroll(env , agent , memory , ob_filter=ob_filter , max_steps=max_steps)
    warmup(env , ob_filter , max_steps=max_steps)
    tf.logging.info('Init training')
    ep = 0
    while ep < max_ep:
        sequence, n_ep = next(seq_gen)
        ep += n_ep
        compute_target(sequence)
        # shuffle is False if recurrent policy
        stats = agent.train(
            Dataset(dict(obs=sequence['obs'] , acts=sequence['acts'] , adv=sequence['adv'] , tdl=sequence['tdl']) ,
                    shuffle=True) , num_iter=10)

        expl_var = explained_variance(sequence['vs'] , sequence['tdl'])
        tf.logging.info('Current ep {},steps so far {} ,explained var {}'.format(ep , ep*max_steps, expl_var))
        tf.logging.info('Avg reward {}, Avg value {}'.format(sequence['rws'].mean() , sequence['vs'].mean()))
        tf.logging.info('\n Policy loss {} \n Value loss {} \n KL {} \n Entropy {} '.format(*stats))



        # TODO dump dataset


def create_env(env_name):
    env = gym.make(env_name)
    env_dims = (
        env.observation_space.shape[0] , env.action_space.shape[0] , (env.action_space.low , env.action_space.high))
    return env , env_dims


def unroll(env , agent , memory , max_steps=2048 , ob_filter=None):
    t = 0
    ob = env.reset()
    ep = 0
    while True:
        if ob_filter: ob = ob_filter(ob)

        act , v = agent.get_action_value(ob)

        if t > 0 and t % max_steps == 0:
            yield memory.release(v=v , done=done), ep

        ob1 , r , done , _ = env.step(act)
        memory.collect((ob , act , r , done , v) , t)
        ob = ob1.copy()
        t += 1
        if done:
            ob = env.reset()
            ep += 1


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
