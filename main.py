import gym
import tensorflow as tf
import threading as th
import multiprocessing
from train import Worker
import os
import time

from agent import Agent

from datetime import datetime

tf.logging.set_verbosity ( tf.logging.INFO )

# 'Walker2d-v1'
# 'Pendulum-v0'

ENV_NAME = 'Walker2d-v1'

# TODO create commands for this as tf.app.flags.DEFINE_string
# TODO when testing on opensim, add 1 more prepprocess layer and split intrnal state (x,y, rotation) with external (dist from obs)

MEMORY_SIZE = 100000
BATCH_SIZE = 64
H_SIZE = [400,300]
GAMMA = 0.99
NUM_EP = 5000
SAVE_EVERY = 100
LOG_DIR = 'logs'
CPU_CORE = multiprocessing.cpu_count()

now = datetime.utcnow ().strftime ( "%b-%d_%H:%M:%S" )  # create unique directories
full_path = os.path.join ( os.getcwd () , 'logs' , now )


def main():
    env = gym.make ( ENV_NAME )

    env_dims = (env.observation_space.shape[ 0 ] ,
                env.action_space.shape[ 0 ] ,
                env.action_space.high)

    workers = [ ]
    threads = [ ]

    # TODO fix this
    # to use a stochastic policy, set stochastic = True, and remove exploration noise from
    # train.py, line 52
    target = Agent ( 'target' , env_dim=env_dims , h_size=H_SIZE, stochastic=False)

    writer = tf.summary.FileWriter ( full_path )
    saver = tf.train.Saver ( tf.get_collection ( tf.GraphKeys.GLOBAL_VARIABLES , scope='target' ) , max_to_keep=2 )
    coord = tf.train.Coordinator ()
    ckpt = tf.train.latest_checkpoint ( full_path )

    # TODO fix this
    # workers.append ( Worker ( env=env , agent=target))

    global_step = tf.Variable ( 0 , trainable=False , name='global_step' )

    for i in range ( CPU_CORE ):
        agent = Agent ( 'worker_{}'.format ( i ) , env_dim=env_dims , target=target , h_size=H_SIZE, writer=writer,stochastic=False)
        env = gym.make ( ENV_NAME )
        workers.append ( Worker ( env=env , agent=agent ) )

    with tf.Session () as sess:

        global_step = tf.Variable ( 0 , trainable=False , name='global_step' )
        sess.run ( tf.global_variables_initializer () )

        if ckpt:
            tf.logging.info ( 'Loading ckpt{}'.format ( ckpt ) )
            saver.restore ( sess , ckpt )

        for w in workers:
            w_fn = lambda: w.run ( sess=sess , coord=coord )
            t = th.Thread ( target=w_fn )
            t.start ()
            threads.append ( t )

        save_th = th.Thread ( target=cont_save ( 1800 , target , coord , saver , writer ) )
        save_th.start ()
        coord.join ( threads )


def cont_save(save_every , agent , coord , saver , writer):
    while not coord.should_stop ():

        sess = tf.get_default_session ()
        env = gym.make ( ENV_NAME )

        global_step = sess.run ( tf.contrib.framework.get_global_step () )

        saver.save ( sess , os.path.join ( full_path , 'model.ckpt' ) , global_step=global_step )

        ep_summary = tf.Summary ()

        state = env.reset ()
        t , rewards = 0 , 0
        terminal = False

        while not terminal:
            env.render ()
            action = agent.get_action ( state )
            state , reward , terminal , _ = env.step ( action )
            rewards += reward
            t += 1
            if terminal:
                break
        env.render ( close=True )

        ep_summary.value.add ( simple_value=rewards , tag='eval/total_rw' )
        ep_summary.value.add ( simple_value=t , tag='eval/ep_length' )

        tf.logging.info (
            'Model saved, current step {}, last rw {} with total steps {}'.format (
                global_step , rewards , t
            )
        )

        writer.add_summary ( ep_summary , global_step )
        writer.flush ()

        time.sleep ( save_every )


if __name__ == '__main__':
    try:
        main ()
    except KeyboardInterrupt:
        tf.train.Coordinator ().request_stop ()
        pass
