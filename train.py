import tensorflow as tf
from datetime import datetime
from agent import Agent
import os
import numpy as np

from utils.ou_noise import OUNoise
from utils.env_wrapper import EnvWrapper
from controller.daddy import Daddy
from ddpg import lrelu
# from motivation.icm import  ICM
# from scale_state import ZFilter


tf.logging.set_verbosity( tf.logging.INFO )

# TODO create argument parser
# TODO create faster way to write experiment readme
# TODO save dataset of experience !
# TODO check prioritized experience code
# TODO rewrite wrapper as FIFO or queue
# TODO Check reward function
# TODO rewrite assisted training

ENV_NAME = 'osim' # 'Walker2d-v1''Pendulum-v0''osim'
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 32
GAMMA = 0.99
NUM_EP = 5000
SAVE_EVERY = 100
H_SIZE = [ 128, 64]
PRE_TRAIN = None
POLICY = 'det' # stochastic, sin
ACTIVATION = lrelu
CONCATENATE_FRAMES  = 3
USE_RW = True
MOTIVATION  = None
CLIP = 20
LOAD_FROM= 'Aug-27_18_06'
NORMALIZE = True

DESCRIPTON = """
Testing Walker2d concatenating 3 frames with ICM model with with reward function. Using prioritzed memory and 128,64, and lrelu. 
Clipping reward and value function.
"""

def main():
    now = datetime.utcnow().strftime( "%b-%d_%H_%M" )  # create unique dir

    full_path = os.path.join( os.getcwd() ,ENV_NAME,'logs' , now )

    if ENV_NAME == 'osim':

        from osim.env import RunEnv

        try:
            env = EnvWrapper(RunEnv, visualize=False,augment_rw=USE_RW, add_time=True, concat=CONCATENATE_FRAMES, normalize=NORMALIZE, add_acceleration = 7)
            split_obs=  env.split_obs
            env_dims = env.get_dims()
        except:
            tf.logging.info('Environment not found')
    else:
        try:
            import gym
            env = gym.make(ENV_NAME)
            env_dims = (env.observation_space.shape[0], env.action_space.shape[0], (env.action_space.low, env.action_space.high))
            split_obs = None
        except:
            raise NotImplementedError

    ou = OUNoise( action_dimension=env_dims[1])

    target = Agent( name='target' , env_dims=env_dims , h_size=H_SIZE , policy = POLICY, act=ACTIVATION, split_obs = split_obs, clip = CLIP)

    global_step = tf.Variable( 0 , trainable=False , name='global_step' )
    writer = tf.summary.FileWriter( full_path )

    with open(os.path.join(full_path,'readme.md'), 'w+') as f:
        f.write(DESCRIPTON)

    # Notice here we are saving only the variables of the target

    agent = Agent( name='local' , env_dims=env_dims , target=target , writer = writer , h_size=H_SIZE ,
                   policy=POLICY, act=ACTIVATION, split_obs=None, motivation = MOTIVATION, clip = CLIP)

    saver = tf.train.Saver( tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES ) , max_to_keep=2 )
    ckpt = tf.train.latest_checkpoint( full_path )

    if PRE_TRAIN is not None:
        daddy = Daddy( target=agent , env_dim=env_dims)

    with tf.Session() as sess:

        if ckpt:
            try:
                tf.logging.info( 'Restore model {}'.format( ckpt ) )
                saver.restore( sess=sess , save_path=ckpt )
            except Exception as e:
                tf.logging.info(e)

        sess.run( tf.global_variables_initializer() )

        if PRE_TRAIN is not None:
            daddy.experience(env = env,epsiodes=PRE_TRAIN, writer=agent.writer)
            saver.save( sess , os.path.join( full_path , 'pre_train', 'model.ckpt' ) , global_step=PRE_TRAIN )
            tf.logging.info('Pre-train ended, starting training now...')

        summarize = False
        ep_summary = tf.Summary()

        for ep in range( NUM_EP ):

            agent.sync()
            state = env.reset()
            ou.reset()

            terminal = False
            timesteps , tot_rw , tot_q , tot_td = 0 , 0 , 0 , 0
            #
            # z_filter = ZFilter( env_dims[0])
            # # state = z_filter( state )

            bonus = 0

            while not terminal:
                # if stochastic remove exploration noisew
                # env.render()

                action = agent.get_action( state ).flatten()  + ou.noise()
                next_state , reward , terminal , _ = env.step( action.flatten() )
                # next_state = z_filter(next_state)

                if MOTIVATION is not None:
                    bonus = agent.motivation.get_bonus(state, next_state, action)
                    bonus += bonus

                td , q = agent.get_td( state , action , reward + bonus, next_state , terminal)

                if CLIP:
                    reward = np.clip(reward, -CLIP, CLIP)

                agent.memory.add( (state , action , reward + bonus, next_state, terminal) , priority=td )
                # agent.memory.add(state, action, reward, next_state, terminal)

                agent.think(summarize=summarize )

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
                    ep_summary.value.add( simple_value=env.r, tag='eval/total_rw' )

                if MOTIVATION:

                    ep_summary.value.add( simple_value=bonus , tag='eval/bonus' )
                    ep_summary.value.add( simple_value=bonus/timesteps , tag='eval/avg_bonus' )

                agent.writer.add_summary( ep_summary , ep )
                agent.writer.flush()

                tf.logging.info(
                    'Master ep  {}, latest avg reward {},bonus taken {},of steps {}'.format( ep , tot_rw / timesteps , bonus,timesteps ) )

            if ep % SAVE_EVERY == 0:
                gs = tf.train.global_step( sess , global_step )
                saver.save( sess , os.path.join( full_path , 'model.ckpt' ) , global_step=gs )
                tf.logging.info( 'Model saved at ep {}'.format( gs ) )




def eval(path, NUM_EP = 1):

    full_path = os.path.join( os.getcwd() , ENV_NAME , 'logs' , path)

    if ENV_NAME == 'osim':

        from osim.env import RunEnv

        try:
            env = EnvWrapper(RunEnv, visualize=True,augment_rw=False, add_time=True, concat=CONCATENATE_FRAMES, normalize=NORMALIZE, add_acceleration = 7)
            env_dims = env.get_dims()
        except:
            tf.logging.info('Environment not found')
    else:
        try:
            import gym
            env = gym.make(ENV_NAME)
            env_dims = (env.observation_space.shape[0], env.action_space.shape[0], (env.action_space.low, env.action_space.high))
        except:
            raise NotImplementedError

    target = Agent( name='target' , env_dims=env_dims , h_size=H_SIZE , policy=POLICY , act=ACTIVATION , split_obs=None , clip=CLIP )

    saver = tf.train.Saver( tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES ) , max_to_keep=2 )
    ckpt = tf.train.latest_checkpoint( full_path )

    with tf.Session() as sess:

        if ckpt:
            try:
                tf.logging.info( 'Restore model {}'.format( ckpt ) )
                saver.restore( sess=sess , save_path=ckpt )
            except Exception as e:
                return  tf.logging.info( e )

        sess.run( tf.global_variables_initializer() )

        for ep in range( NUM_EP ):

            state = env.reset()
            timesteps , tot_rw , tot_q , tot_td = 0 , 0 , 0 , 0

            terminal  = False

            while not terminal:

                if ENV_NAME !='osim':

                    env.render()

                action = target.get_action( state ).flatten()
                next_state , reward , terminal , _ = env.step(action )
                state = next_state
                timesteps += 1
                tot_rw += reward

            tf.logging.info(
                'Master ep  {}, latest avg reward {},of steps {}'.format( ep , tot_rw / timesteps, timesteps ) )
    env.close()


if __name__ == '__main__':
    if LOAD_FROM is not None:
        eval(LOAD_FROM)
    else:
        main()