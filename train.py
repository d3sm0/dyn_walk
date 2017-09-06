import tensorflow as tf
import json

from memory.dataset import Dataset
from utils.logger import Logger
from utils.running_stats import ZFilter

from worker import Worker

tf.logging.set_verbosity(tf.logging.INFO)


# remember with no trustd region you tested 128 256 512, wuth trusted region you tested 512 256 128
def main(config):
    logger = Logger(env_name=config['ENV_NAME'])
    worker = Worker(config , log_dir=logger.main_path)

    ob_filter = None
    if config['ENV_NAME'] != 'osim':
        ob_filter = ZFilter((worker.env_dim[0] ,))

    play(worker , config , logger=logger , ob_filter=ob_filter)


def play(worker , config , logger=None , ob_filter=None):
    worker.warmup(ob_filter , max_steps=config['MAX_STEPS'])

    seq_gen = worker.unroll(ob_filter=ob_filter)

    tf.logging.info('Init training')

    t = 0
    while t < config['MAX_STEPS'] * config['MAX_EP']:
        sequence , ep_stats = next(seq_gen)

        worker.compute_target(sequence)

        b = Dataset(dict(obs=sequence['obs'] , acts=sequence['acts'] , adv=sequence['adv'] , tdl=sequence['tdl'] ,
                         vs=sequence['vs']) , batch_size=config['BATCH_SIZE'] , shuffle=True)

        stats = worker.agent.train(b , num_iter=config['NUM_ITER'] , eps=config['EPS'])

        logger.log(stats)
        logger.log(ep_stats)
        if t % config['REPORT_EVERY'] == 0:
            logger.write(display=True)
            worker.write_summary(ep_stats, ep_stats['ep'])
        if t % config['SAVE_EVERY'] == 0:
            worker.agent.save(log_dir=logger.main_path)
            tf.logging.info('Saved model at ep {}'.format(ep_stats['ep']))
        t = ep_stats['t']

if __name__ == '__main__':
    with open('config.json') as f:
        kwargs = json.load(f)

    main(kwargs)
