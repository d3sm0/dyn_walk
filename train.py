import tensorflow as tf

from datetime import datetime
import os
import numpy as np

from agent import Agent
from utils.ou_noise import OUNoise
from utils.env_wrapper import EnvWrapper

from ddpg import lrelu

tf.logging.set_verbosity( tf.logging.INFO )

# 'Walker2d-v1'
# 'Pendulum-v0'

# TODO create argument parser
# TODO create faster way to write experiment readme
# TODO save dataset of experience !
# TODO check prioritized experience code
# TODO rewrite wrapper as FIFO or queue

ENV_NAME = 'osim'  # 'Walker2d-v1''Pendulum-v0''osim'
MEMORY_SIZE = int( 1e6 )
BATCH_SIZE = 32
GAMMA = 0.99
NUM_EP = 5000
SAVE_EVERY = 100
H_SIZE = [ 128 , 64 ]
PRE_TRAIN = None
POLICY = 'det'  # stochastic, sin
ACTIVATION = lrelu
CONCATENATE_FRAMES = 3
USE_RW = True
MOTIVATION = None
CLIP = 20
LOAD_FROM = None # 'Aug-29_12_08' # 'Aug-29_12_08' # 'Aug-28_22_29'  # 'Aug-27_18_06'
FRAME_RATE  = 50 # pick 1/4
NORMALIZE = True # Recenter wrt to the torso and Statistically normalization
DESCRIPTON = 'Testing osim concatenatig 3 frames with with augmented reward, batch normalization with running mean and variance.' \
             'Using clipping gradients and value function. Using prioritzed memory and 128,64 as size of hidden layer and shared first two layers' \
             'Testing tanh activation function and picking 1 over 4 frames.'

def main():

    now = datetime.utcnow().strftime( "%b-%d_%H_%M" )  # create unique dir

    # now = 'Aug-29_12_08'
    full_path = os.path.join( os.getcwd() , ENV_NAME,'logs' , now )

    if ENV_NAME == 'osim':

        from osim.env import RunEnv

        try:
            env = EnvWrapper( RunEnv , visualize=False , augment_rw=USE_RW , add_time=False ,
                              concat=CONCATENATE_FRAMES , normalize = NORMALIZE, add_acceleration=7, frame_rate=FRAME_RATE)
            split_obs = env.split_obs
            env_dims = env.get_dims()
        except:
            tf.logging.info( 'Environment not found' )
    else:
        try:
            import gym
            env = gym.make( ENV_NAME )
            env_dims = (env.observation_space.shape[ 0 ] , env.action_space.shape[ 0 ] ,
                        (env.action_space.low , env.action_space.high))
            split_obs = None
        except:
            raise NotImplementedError

    global_step = tf.Variable( 0 , trainable=False , name='global_step' )
    writer = tf.summary.FileWriter( full_path )

    ckpt = tf.train.latest_checkpoint( full_path )

    target = Agent( name='target' , env_dims=env_dims , h_size=H_SIZE , policy=POLICY , act=ACTIVATION ,
                    split_obs=None )
    agent = Agent( name='local' , env_dims=env_dims , target=target , writer=writer , h_size=H_SIZE ,
                   policy=POLICY , act=ACTIVATION , split_obs=None )

    with open( os.path.join( full_path , 'readme.md' ) , 'w+' ) as f:
        f.write( DESCRIPTON )

    with tf.Session() as sess:
        #
        # tf.logging.info( 'Restore model {}'.format( ckpt ) )
        # saver.restore( sess=sess , save_path=ckpt )
        #
        # # if ckpt:
        # #     try:
        # #
        # #     except Exception as e:
        # #         tf.logging.info( e )

        sess.run( tf.global_variables_initializer() )

        saver = tf.train.Saver( tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES ) , max_to_keep=2 )

        summarize = False
        ep_summary = tf.Summary()
        ou = OUNoise( action_dimension=env_dims[ 1 ] )

        for ep in range( NUM_EP ):

            agent.sync()
            state = env.reset()
            ou.reset()

            terminal = False
            timesteps , tot_rw , tot_q , tot_td = 0 , 0 , 0 , 0

            while not terminal:
                # if stochastic remove exploration noise

                action = agent.get_action( state ).flatten() + ou.noise()
                next_state , reward , terminal , _ = env.step( action )

                if CLIP:
                    reward = np.clip( reward , -CLIP , CLIP )

                td , q = agent.get_td( state , action , reward , next_state , terminal )
                agent.memory.add( (state , action , reward , next_state , terminal) , priority=td )

                agent.think( summarize=summarize )

                state = next_state
                summarize = False
                timesteps += 1

                tot_td += td
                tot_q += q
                tot_rw += reward

            if ep % 5 == 0:

                summarize = True

                ep_summary.value.add( simple_value=tot_rw / timesteps , tag='eval/avg_surr_rw' )
                ep_summary.value.add( simple_value=tot_q / timesteps , tag='eval/avg_q' )
                ep_summary.value.add( simple_value=tot_td / timesteps , tag='eval/avg_td' )
                ep_summary.value.add( simple_value=tot_rw , tag='eval/surr_rw' )
                ep_summary.value.add( simple_value=tot_td , tag='eval/total_td' )
                ep_summary.value.add( simple_value=tot_q , tag='eval/total_q' )
                ep_summary.value.add( simple_value=timesteps , tag='eval/ep_length' )

                if ENV_NAME == 'osim':
                    ep_summary.value.add( simple_value=env.r / timesteps , tag='eval/avg_rw' )
                    ep_summary.value.add( simple_value=env.r , tag='eval/total_rw' )

                agent.writer.add_summary( ep_summary , ep )
                agent.writer.flush()

                tf.logging.info(
                    'Master ep  {}, latest avg reward {}, of steps {}'.format( ep , tot_rw / timesteps , timesteps ) )

            if ep % SAVE_EVERY == 0:
                gs = tf.train.global_step( sess , global_step )
                saver.save( sess , os.path.join( full_path , 'model.ckpt' ) , global_step=gs )
                tf.logging.info( 'Model saved at ep {}'.format( gs ) )


def eval(path , NUM_EP=5):
    full_path = os.path.join( os.getcwd() , ENV_NAME , 'logs' , path )

    if ENV_NAME == 'osim':

        from osim.env import RunEnv

        try:
            env = EnvWrapper( RunEnv , visualize=True , augment_rw=False , add_time=False , concat=CONCATENATE_FRAMES ,
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

    saver = tf.train.Saver( tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES ) , max_to_keep=2 )
    ckpt = tf.train.latest_checkpoint( full_path )

    with tf.Session() as sess:

        if ckpt:
            tf.logging.info( 'Restore model {}'.format( ckpt ) )
            saver.restore( sess=sess , save_path=ckpt )


        # sess.run( tf.global_variables_initializer() )

        for ep in range( NUM_EP ):

            state = env.reset()
            timesteps , tot_rw , tot_q , tot_td = 0 , 0 , 0 , 0

            terminal = False

            while not terminal:

                if ENV_NAME != 'osim':
                    env.render()

                action = target.get_action( state ).flatten()
                # print(action)
                next_state , reward , terminal , _ = env.step( action )
                state = next_state
                timesteps += 1
                tot_rw += reward

            tf.logging.info(
                'Master ep  {}, latest avg reward {},of steps {}'.format( ep , tot_rw / timesteps , timesteps ) )
    env.close()


if __name__ == '__main__':
    if LOAD_FROM is not None:
        eval( LOAD_FROM )
    else:
        main()
