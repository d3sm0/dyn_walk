import tensorflow as tf

from datetime import datetime
from agent import Agent
from ou_noise import OUNoise
# from daddy import Daddy, augment_state
import os
import gym

# from osim.env import RunEnv
import numpy as np

tf.logging.set_verbosity( tf.logging.INFO )

# 'Walker2d-v1'
# 'Pendulum-v0'

ENV_NAME = 'Walker2d-v1'

MEMORY_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
NUM_EP = 5000
SAVE_EVERY = 100
H_SIZE = [ 128 , 128 ]
PRE_TRAIN = 100
IS_STOCHASTIC = True

def main():
    now = datetime.utcnow().strftime( "%b-%d_%H_%M" )  # create unique dir

    full_path = os.path.join( os.getcwd() , 'logs' , now )

    env = gym.make( ENV_NAME )

    # env = RunEnv( visualize=False )
    # 5 is the number of velocities for head and other parts
    env_dims = (env.observation_space.shape[ 0 ], env.action_space.shape[ 0 ] , (env.action_space.low, env.action_space.high))
    ou = OUNoise( action_dimension=env_dims[ 1 ] )

    # tf.reset_default_graph ()

    target = Agent( name='target' , env_dim=env_dims , h_size=H_SIZE, memory_size=MEMORY_SIZE,stochastic=IS_STOCHASTIC )

    global_step = tf.Variable( 0 , trainable=False , name='global_step' )

    writer = tf.summary.FileWriter( full_path )
    saver = tf.train.Saver( tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , scope='target' ) , max_to_keep=2 )
    ckpt = tf.train.latest_checkpoint( full_path )

    agent = Agent( name='local' , env_dim=env_dims , target=target , writer=writer , h_size=H_SIZE , stochastic=IS_STOCHASTIC )

    # daddy = Daddy( target=agent , env_dim=env_dims )

    with tf.Session() as sess:
        if ckpt:
            tf.logging.info('Restore model {}'.format(ckpt))
            saver.restore(sess=sess,  save_path=ckpt)

        sess.run( tf.global_variables_initializer() )

        summarize = False
        # load pre trained model
        #
        # for _ in range(PRE_TRAIN):
        #     l, tot_rw, timesteps = daddy.sample(env= env, w = daddy.w)
        #
        #     if _ % 5 == 0:
        #         ep_summary = tf.Summary()
        #
        #         ep_summary.value.add( simple_value=tot_rw , tag='daddy/total_rw' )
        #         ep_summary.value.add( simple_value=timesteps , tag='daddy/timesteps' )
        #         ep_summary.value.add( simple_value=l , tag='daddy/loss' )
        #
        #         agent.writer.add_summary( ep_summary , _ )
        #         agent.writer.flush()
        #
        #         tf.logging.info(
        #             'Master ep  {}, latest ep reward {}, of steps {}'.format( _ , tot_rw , timesteps ) )
        #
        # tf.logging.info('Pre-train ended, starting training now...')
        # # save memory
        # daddy.save_memory('memory.pkl')
        # # transfer memory
        # agent.memory.buffer = copy.deepcopy(daddy.memory.buffer)
        #
        # saver.save( sess , os.path.join( full_path , 'model.ckpt' ) , global_step=PRE_TRAIN )

        for ep in range( NUM_EP ):

            agent.sync()
            state = env.reset()
            ou.reset()

            terminal = False

            timesteps , tot_rw = 0 , 0


            # activate if osim_rl
            # state = augment_state( state , state )

            while not terminal:
                # if stochastic remove exploration noise
                action = agent.get_action( state )  # + ou.noise()

                # if determinstic clip here
                # action = np.clip(action, 0, 1)

                next_state , reward , terminal , _ = env.step( action.flatten() )

                # Activate if osim-rl
                # rw = surr_rw( state , action ) + reward
                # next_state = augment_state( state , next_state )

                agent.memory.collect( state , action , reward , next_state , terminal )
                agent.think( batch_size=BATCH_SIZE , gamma=GAMMA , summarize=summarize )

                state = next_state
                summarize = False

                timesteps += 1
                tot_rw += reward

            if ep % 5 == 0:
                summarize = True
                ep_summary = tf.Summary()

                ep_summary.value.add( simple_value=tot_rw , tag='eval/total_rw' )
                ep_summary.value.add( simple_value=timesteps , tag='eval/ep_length' )

                agent.writer.add_summary( ep_summary , ep )
                agent.writer.flush()

                tf.logging.info(
                    'Master ep  {}, latest ep reward {}, of steps {}'.format( ep , tot_rw , timesteps ) )

            if ep % SAVE_EVERY == 0:
                gs = tf.train.global_step( sess , global_step )
                saver.save( sess , os.path.join( full_path , 'model.ckpt' ) , global_step=gs )
                tf.logging.info( 'Model saved at ep {}'.format( gs ) )


def augment_state(s , s1):

    s = np.reshape( np.array( s ) , (1 , -1) )
    s1 = np.reshape( np.array( s1 ) , (1 , -1) )

    idxs = [ 22 , 24 , 26 , 28 , 30 ]

    vel = (s1[ : , idxs ] - s[ : , idxs ]) / (0.01)
    # keep information of the environment like difficulty
    return np.reshape( np.append( s1[:,:38] , np.append(vel, s1[:,38:])) , (1 , -1) )


def surr_rw(state , action):
    # state = np.array(state)
    delta_h = state[ :,27 ] - state[ :,35 ]
    rw = 10 * state[ :,20 ] - abs( delta_h - 1.2 ) - 0.1 * np.linalg.norm( action ) - 10 * (state[ :,27 ] < 0.8)
    return np.asscalar( rw )


if __name__ == '__main__':
    main()
