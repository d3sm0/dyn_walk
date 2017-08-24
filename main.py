import gym
from gym.wrappers import Monitor
import tensorflow as tf
import threading as th
import multiprocessing
from train import Worker
import os
import time

from agent import Agent

from datetime import datetime

tf.logging.set_verbosity(tf.logging.INFO)

# 'Walker2d-v1'
# 'Pendulum-v0'

"""
TODO:
- fix gradient update on mpi
- make it ready for osim simulation
- create variant with lstm
"""

ENV_NAME = 'Walker2d-v1'

# TODO create commands for this as tf.app.flags.DEFINE_string
# TODO when testing on opensim, add 1 more prepprocess layer and split intrnal state (x,y, rotation) with external (dist from obs)
# TODO Test with batch norm

MEMORY_SIZE = 10e3  # 10e6
BATCH_SIZE = 64  # 64
H_SIZE = [256, 128]  # [400, 300] in the orginal paper
GAMMA = 0.99
NUM_EP = 10000
SAVE_EVERY = 1800
LOG_DIR = 'logs'
# multi thread does not scale
# CPU_CORE = multiprocessing.cpu_count()
CPU_CORE = 1
# now = datetime.utcnow ().strftime ( "%b-%d_%H_%M" )  # create unique dir
now = 'Aug-18_20_25'
full_path = os.path.join(os.getcwd(), 'logs', now)


def main():
    env = gym.make(ENV_NAME)

    env_dims = (env.observation_space.shape[0],
                env.action_space.shape[0],
                env.action_space.high)

    workers = []
    processes = []

    # TODO fix this
    # to use a stochastic policy, set stochastic = True, and remove exploration noise from
    # train.py, line 52
    target = Agent('target', env_dim=env_dims, h_size=H_SIZE, stochastic=True)

    writer = tf.summary.FileWriter(logdir=full_path)
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target'), max_to_keep=2)
    coord = tf.train.Coordinator()
    ckpt = tf.train.latest_checkpoint(full_path)

    # TODO fix this
    # workers.append ( Worker ( env=env , agent=target))

    global_step = tf.Variable(0, trainable=False, name='global_step')

    """
    # start N processes
    #   do stuff
    # after a while restart process
    """
    NR_WORKERS = 1000
    for i in range(NR_WORKERS):
        agent = Agent('worker_{}'.format(i), env_dim=env_dims, target=target,
                      h_size=H_SIZE, writer=writer, stochastic=True)
        workers.append(Worker(agent=agent))

    with tf.Session() as sess:
        global_step = tf.Variable(0, trainable=False, name='global_step')
        sess.run(tf.global_variables_initializer())

        if ckpt:
            tf.logging.info('Loading ckpt{}'.format(ckpt))
            saver.restore(sess, ckpt)

        def run_worker(worker, sess, coord):
            worker.run(sess=sess, coord=coord)

        with multiprocessing.Pool(processes=CPU_CORE) as pool:
            pool.map(run_worker, zip(workers, NR_WORKERS*[sess], NR_WORKERS*[coord]))

        save_th = th.Thread(target=cont_save(SAVE_EVERY, target, coord, saver, writer))
        save_th.start()
        # save_th.join()
        coord.join(threads)


def cont_save(save_every, agent, coord, saver, writer):
    while not coord.should_stop():

        sess = tf.get_default_session()
        env = gym.make(ENV_NAME)

        global_step = sess.run(tf.train.get_global_step())

        saver.save(sess, os.path.join(full_path, 'model.ckpt'), global_step=global_step)

        # env = Monitor(env, full_path, force=True)

        state = env.reset()
        t, rewards = 0, 0
        terminal = False

        while not terminal:
            env.render()
            action = agent.get_action(state)
            state, reward, terminal, _ = env.step(action)
            rewards += reward
            t += 1
            if terminal:
                break
        env.render(close=True)
        time.sleep(save_every)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        tf.train.Coordinator().request_stop()
        pass
