import tensorflow as tf
from agent import Worker, Agent, Target

from datetime import datetime
import os
import numpy as np

from agent import Agent
from utils.env_wrapper import EnvWrapper

from ddpg import lrelu
from motivation.icm import ICM
import argparse
import json


tf.logging.set_verbosity(tf.logging.INFO)


# TODO create argument parser
# TODO create faster way to write experiment readme
# TODO save dataset of experience !
# TODO check prioritized experience code
# TODO rewrite wrapper as FIFO or queue
# TODO Check reward function


def main(config):
    now = datetime.utcnow().strftime("%b-%d_%H_%M")  # create unique dir
    log_dir = os.path.join(os.getcwd(), config['ENV_NAME'], 'logs', now)
    env, env_dims, split_obs = Worker.init_environment(config)
    env.close()
    global_step = tf.Variable(0, trainable=False, name='global_step')
    try:
        os.makedirs(log_dir)
    except:
        pass

    with open(os.path.join(log_dir, 'readme.md'), 'a') as f:
        f.write(config['DESCRIPTON'])

    target = Target(log_dir)

    target.initialize(env_dims=env_dims,
                      h_size=config['H_SIZE'], policy=config['POLICY'], act=eval(config['ACTIVATION']))
    worker = Worker(target, config, log_dir)

    # worker does stuff, a parallel worker
    for ep in range(config['NUM_EP']):
        worker.agent.update_target_network()

        tot_td, tot_q, tot_rw, timesteps = worker.unroll()

        if ep % config['REPORT_EVERY'] == 0:
            worker.report_metrics(tot_td, tot_q, tot_rw, timesteps, ep)

        if ep % config['SAVE_EVERY'] == 0:
            worker.agent.save_progress()

        tf.logging.info('Master ep  {}, latest avg reward {}, of steps {}'.format(ep, tot_rw / timesteps, timesteps))


if __name__ == '__main__':
    with open("config.json") as fin:
        kwargs = json.load(fin)
    main(kwargs)
