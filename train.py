import tensorflow as tf
from agent import Target
from worker import Worker

from datetime import datetime
import os

import json

tf.logging.set_verbosity(tf.logging.INFO)


# TODO create argument parser
# TODO create faster way to write experiment readme
# TODO save dataset of experience !


def main(config):
    now = datetime.utcnow().strftime("%b-%d_%H_%M")  # create unique dir
    log_dir = os.path.join(os.getcwd() , config['ENV_NAME'] , 'logs' , now)
    env , env_dims , split_obs = Worker.init_environment(config)
    env.close()
    global_step = tf.Variable(0 , trainable=False , name='global_step')
    try:
        os.makedirs(log_dir)
    except:
        pass

    with open(os.path.join(log_dir , 'readme.md') , 'a') as f:
        f.write(config['DESCRIPTON'])

    target = Target(log_dir)

    target.initialize(env_dims=env_dims ,
                      h_size=config['H_SIZE'] , policy=config['POLICY'] , act=eval(config['ACTIVATION']))
    worker = Worker(target , config , log_dir)

    # warmup to adjust running stats
    worker.warmup(ep = 5)

    # worker does stuff, a parallel worker
    for ep in range(config['NUM_EP']):
        worker.agent.update_target_network()

        tot_td , tot_q , tot_rw , timesteps = worker.unroll(curr_ep = ep)

        if ep % config['REPORT_EVERY'] == 0:
            worker.report_metrics(tot_td , tot_q , tot_rw , timesteps , ep)
            tf.logging.info(
                'Master ep  {}, latest avg reward {}, of steps {}'.format(ep , tot_rw / timesteps , timesteps))

        if ep % config['SAVE_EVERY'] == 0:
            worker.agent.save_progress()

def eval(path , NUM_EP=5):
    full_path = os.path.join( os.getcwd() , ENV_NAME , 'logs' , path )

    if ENV_NAME == 'osim':

        from osim.env import RunEnv

        try:
            env = EnvWrapper( RunEnv , visualize=True , augment_rw=True , add_time=False , concat=CONCATENATE_FRAMES ,
                              normalize=NORMALIZE , add_acceleration=7 )
            env_dims = env.get_dims()
        except:
            tf.logging.info( 'Environment not found' )
    else:
        try:
            import gym
            env = gym.make( ENV_NAME )
            env_dims = (env.observation_space.shape[ 0 ] , env.action_space.shape[ 0 ] ,
                        (env.action_space.low , env.action_space.high))
        except:
            raise NotImplementedError

    target = Agent( name='target' , env_dims=env_dims , h_size=H_SIZE , policy=POLICY , act=ACTIVATION ,
                    split_obs=None , clip=CLIP )

    agent = Agent( name='local' , env_dims=env_dims , target=target , h_size=H_SIZE ,
                   policy=POLICY , act=ACTIVATION , split_obs=None )

    with tf.Session() as sess:

        # sess.run( tf.global_variables_initializer() )

        saver = tf.train.Saver( tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES ) , max_to_keep=2 )
        ckpt = tf.train.latest_checkpoint( full_path )

        if ckpt:
            try:
                tf.logging.info( 'Restore model {}'.format( ckpt ) )
                saver.restore( sess=sess , save_path=ckpt )
            except Exception as e:
                tf.logging.info( e )
                raise Exception

        for ep in range( NUM_EP ):

            state = env.reset()
            timesteps , tot_rw , tot_q , tot_td = 0 , 0 , 0 , 0

            terminal = False

            while not terminal:

                if ENV_NAME != 'osim':
                    env.render()

                action = agent.get_action( state ).flatten()

                next_state , reward , terminal , _ = env.step( action )
                # print(reward)
                state = next_state
                timesteps += 1
                tot_rw += reward

            tf.logging.info(
                'Master ep  {}, latest avg reward {},of steps {}'.format( ep , tot_rw / timesteps , timesteps ) )
        env.close()


if __name__ == '__main__':
    with open("config.json") as fin:
        kwargs = json.load(fin)
    main(kwargs)
