import json

import tensorflow as tf

from memory.dataset import Dataset
# TODO check logger it seems slow
from utils.logger import Logger
from utils.misc_utils import merge_dicts
from utils.running_stats import ZFilter
from worker import Worker
from pprint import pprint

tf.logging.set_verbosity(tf.logging.INFO)

tf.set_random_seed(10)
import numpy as np
import random
np.random.seed(10)
random.seed(10)
# TODO check for topology (256, 128,64, 32) (256,128,64)

def main(config):
    pprint(config)
    logger = Logger(env_name=config['ENV_NAME'], config=config)
    logger.save_experiment(config)
    worker = Worker(config, log_dir=logger.main_path)
    # meta_graph_def = tf.train.export_meta_graph(filename=logger.main_path + '/model.meta')
    ob_filter = None
    if config['ENV_NAME'] != 'osim':
        ob_filter = ZFilter((worker.env_dim[0],))
    worker.warmup(ob_filter, max_steps=config['WARMUP_TIME'], history_depth=config['CONCATENATE_FRAMES'])
    tf.logging.info('Init training. Stats saved at ' + logger.main_path)
    # Sep-23_12_38
    # oldpi , oldv = worker.agent.sess.run([worker.agent.policy._params , worker.agent.value._params])
    t = 0
    unrolls = 0
    # dataset_path = None #'log-files/InvertedPendulum-v1/dataset'  # 'log-files/InvertedPendulum-v1/Sep-22_13_29'

    dataset_path = 'log-files/Walker2d-v1/dataset' #'log-files/InvertedPendulum-v1/dataset'
    worker.imagination.load(data_dir=dataset_path)

    while t < config['MAX_STEPS']:
        sequence, ep_stats = worker.unroll(max_steps=config['MAX_STEPS_BATCH'], ob_filter=ob_filter,
                                           n_branches=config['LOG_BRANCH_WIDTH'],
                                           history_depth=config['CONCATENATE_FRAMES'])

        batch = worker.compute_target(sequence)

        dataset = Dataset(dict(obs=batch['obs'], acts=batch['acts'], adv=batch['adv'], tdl=batch['tdl'],
                               vs=batch['vs']), batch_size=config['BATCH_SIZE'],
                          shuffle=False if config['POLICY'] == 'recurrent' else True)

        train_stats, network_stats = worker.agent.train(dataset, num_iter=config['NUM_ITER'], eps=config['EPS'])
        model_stats = worker.imagination.train(shuffle=False if config['IMG_MODEL'] == 'recurrent' else True)

        logger.log(merge_dicts(train_stats, ep_stats, model_stats))
        # if t % config['REPORT_EVERY'] == 0:
        logger.write(display=True)
        worker.write_summary(merge_dicts(ep_stats, train_stats), ep_stats['total_steps'], network_stats=network_stats)
        if t % config['SAVE_EVERY'] == 0:
            worker.agent.save(save_dir=logger.main_path)
            tf.logging.info('Saved model at ep {}'.format(ep_stats['total_steps']))

        t += ep_stats['total_steps']
        unrolls += 1


if __name__ == '__main__':
    with open('config.json') as f:
        kwargs = json.load(f)

    main(kwargs)