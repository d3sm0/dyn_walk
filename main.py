import json

import tensorflow as tf

from memory.dataset import Dataset
from utils.logger import Logger
from utils.misc_utils import merge_dicts
from utils.running_stats import ZFilter
from worker import Worker
from pprint import pprint
from utils.tf_utils import set_global_seed

tf.logging.set_verbosity(tf.logging.INFO)


def main(config):
    pprint(config)
    set_global_seed(10)
    logger = Logger(env_name=config['ENV_NAME'], config=config)
    logger.save_experiment(config)
    worker = Worker(config, log_dir=logger.main_path)
    ob_filter = ZFilter((worker.env_dim[0],))
    worker.warmup(ob_filter, max_steps=config['WARMUP_TIME'])
    tf.logging.info('Init training. Stats saved at ' + logger.main_path)
    t = 0
    unrolls = 0

    while t < config['MAX_STEPS']:
        sequence, ep_stats = worker.unroll(max_steps=config['MAX_STEPS_BATCH'], ob_filter=ob_filter)

        batch = worker.compute_target(sequence)

        dataset = Dataset(dict(obs=batch['obs'], acts=batch['acts'], adv=batch['adv'], tdl=batch['tdl'],
                               vs=batch['vs']), batch_size=config['BATCH_SIZE'],
                          shuffle=False if config['POLICY'] == 'recurrent' else True)

        train_stats, network_stats = worker.agent.train(dataset, num_iter=config['NUM_ITER'], eps=config['EPS'])

        logger.log(merge_dicts(train_stats, ep_stats))
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
