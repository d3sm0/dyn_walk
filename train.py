import tensorflow as tf
import json

from memory.dataset import Dataset
# TODO check logger it seems slow
from utils.logger import Logger
from utils.misc_utils import merge_dicts
from utils.running_stats import ZFilter

from worker import Worker

tf.logging.set_verbosity(tf.logging.INFO)

# TODO check for topology (256, 128,64, 32) (256,128,64)



def main(config):


    logger = Logger(env_name=config['ENV_NAME'])
    logger.save_experiment(config)
    worker = Worker(config , log_dir=logger.main_path)
    ob_filter = None
    if config['ENV_NAME'] != 'osim':
        ob_filter = ZFilter((worker.env_dim[0] ,))

    play(worker , config , logger=logger , ob_filter=ob_filter)



def play(worker , config , logger=None , ob_filter=None):
    worker.warmup(ob_filter , max_steps=config['WARMUP_TIME'])

    seq_gen = worker.unroll(ob_filter=ob_filter , max_steps=config['MAX_STEPS_BATCH'])

    tf.logging.info('Init training. Stats saved at ' + logger.main_path)

    t = 0
    while t < config['MAX_STEPS']:
        sequence , ep_stats = next(seq_gen)

        batch = worker.compute_target(sequence)
        # # TODO check here how to update properly. Batch is a deep copy of sequence
        # obs , acts , adv , tdl , vs = batch['obs'] , batch['acts'] , batch['adv'] , batch['tdl'] , batch['vs']
        # adv = (adv - adv.mean()) / adv.std()
        # # b = Dataset(dict(obs=batch['obs'] , acts=batch['acts'], adv=adv, tdl=batch['tdl'] ,
        # #                  vs=batch['vs']) , batch_size=config['BATCH_SIZE'] , shuffle=True)

        # obs, acts, adv, tdl, vs = batch['obs'], batch['acts'], batch['adv'], batch['tdl'], batch['vs']
        #
        dataset = Dataset(dict(obs=batch['obs'] , acts=batch['acts'] , adv=batch['adv'] , tdl=batch['tdl'] ,
                               vs=batch['vs']) , batch_size=config['BATCH_SIZE'] , shuffle=True)

        # dataset = Dataset(dict(obs=obs , acts=acts , adv=obs , tdl=tdl ,
        #                        vs=vs) , batch_size=config['BATCH_SIZE'] , shuffle=True)

        train_stats, network_stats = worker.agent.train(dataset , num_iter=config['NUM_ITER'] , eps=config['EPS'])

        logger.log(merge_dicts(train_stats , ep_stats))
        # if t % config['REPORT_EVERY'] == 0:
        logger.write(display=True)
        worker.write_summary(merge_dicts(ep_stats , train_stats) , ep_stats['total_ep'] , network_stats=network_stats)
        if t % config['SAVE_EVERY'] == 0:
            worker.agent.save(log_dir=logger.main_path)
            tf.logging.info('Saved model at ep {}'.format(ep_stats['total_ep']))

        t = ep_stats['total_steps']


if __name__ == '__main__':
    with open('config.json') as f:
        kwargs = json.load(f)

    main(kwargs)
