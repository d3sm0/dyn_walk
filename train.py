import gym
from dataset import Dataset
import numpy as np

env = gym.make('Walker2d-v1')
def get_trajectory(max_steps=2048):
    t = 0
    action = env.action_space.sample()
    done = True
    state = env.reset()
    states = np.array([state for _ in range(max_steps)])
    actions = np.array([action for _ in range(max_steps)])
    rewards = np.zeros(max_steps , 'float32')
    values = np.zeros(max_steps , 'float32')
    dones = np.zeros(max_steps , 'int32')

    while True:
        last_action = a
        a = env.action_space.sample()
        v_pred = 2
        if t > 0 and t % max_steps == 0:
            yield {
                'obs': obs ,
                'acts': acts ,
                'rws': rws ,
                'vs': vs ,
                'vs_next': v_pred * (1 - done) ,
                'ds': ds
            }

        i = t % max_steps
        obs[i] = ob
        acts[i] = a
        vs[i] = v_pred
        ob , r , done , _ = env.step(a)
        ds[i] = done
        if t:
            ob = env.reset()
        t += 1


def compute_target(seq , gamma=0.99 , lam=1.0):
    dones = np.append(seq['ds'] , 0)
    v_hat = np.append(seq['vs'] , seq['vs_next'])
    T = len(seq['rws'])
    seq['adv'] = gae = np.empty(T , 'float32')
    rws = seq['rws']
    last_gae = 0
    for t in reversed(range(T)):
        not_terminal = 1 - dones[t + 1]
        # td error
        delta = rws[t] + 0.99 * v_hat[t + 1] * not_terminal - v_hat[t]
        gae[t] = last_gae = delta + 0.99 * 1.0 * not_terminal * last_gae
    seq['tdl'] = seq['adv'] + seq['vs']
