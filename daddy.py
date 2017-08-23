import numpy as np
import pickle
import tensorflow as tf
from replay_buffer import ReplayBuffer

file_name = 'w.pkl'


class Daddy( object ):
    def __init__(self , target , env_dim, buffer_size = 1e6):
        self.target = target
        self.action = tf.placeholder( 'float32' , shape=[ None , env_dim[1] ] , name='daddy_act' )
        self.loss = tf.reduce_mean( tf.squared_difference( self.action , target.actor.mu_hat ) )
        self.train_step = tf.train.AdamOptimizer().minimize( self.loss )
        # transfer update to critic
        self.sync = target.update_target( target.actor.params , target.critic.params[ :8 ] ,
                                          tau=1 )
        self.w = self.load_w( file_name=file_name )
        self.memory = ReplayBuffer(env_shape = env_dim, buffer_size=buffer_size)
        self.T = 4

    def teach(self , state , action):
        sess = tf.get_default_session()
        l , _ = sess.run( [ self.loss , self.train_step ] ,
                          feed_dict={self.target.actor.state: state , self.action: action} )
        return l

    def sync(self):
        sess = tf.get_default_session()
        sess.run( self.sync )

    def dump_model(self , file_name , w):
        with open( file_name , 'wb' ) as f:
            pickle.dump( w , f )
            print('Model dumped')

    def output(self , a , t):
        y = 0

        for i in range( 4 ):
            y += a[ i ] * np.sin( (i + 1) * np.pi * 2 * t / self.T + a[ i + 4 ] )
        return y

    def get_action(self , w , t):

        inputs = [ -self.output( w[ 0 ] , t ) , self.output( w[ 1 ] , t ) , -self.output( w[ 2 ] , t ) ,
                   self.output( w[ 3 ] , t ) , self.output( w[ 4 ] , t ) , self.output( w[ 5 ] , t ) ,
                   -self.output( w[ 6 ] , t ) , -self.output( w[ 7 ] , t ) , self.output( w[ 8 ] , t ) ,
                   self.output( w[ 0 ] , t + self.T / 2 ) , -self.output( w[ 1 ] , t + self.T / 2 ) ,
                   self.output( w[ 2 ] , t + self.T / 2 ) ,
                   -self.output( w[ 3 ] , t + self.T / 2 ) , -self.output( w[ 4 ] , t + self.T / 2 ) ,
                   -self.output( w[ 5 ] , t + self.T / 2 ) ,
                   self.output( w[ 6 ] , t + self.T / 2 ) , self.output( w[ 7 ] , t + self.T / 2 ) ,
                   -self.output( w[ 8 ] , t + self.T / 2 ) , ]

        return inputs

    def load_w(self , file_name):
        try:
            with open( file_name , 'rb' ) as f:
                w = pickle.load( f )
            return w
        except IOError:
            print('File not found')

    def save_memory(self, file_name):
        try:
            with open(file_name, 'wb') as f:
                pickle.dump(self.memory.buffer, f)
                tf.logging.info('Saved memory')
        except Exception as e:
            tf.logging.info(e)

    def sample(self , env , w):
        diff = np.random.randint( 2 )
        seed = np.random.randint( 200 )
        state = env.reset( difficulty=diff , seed=seed )
        tf.logging.info('Starting new env with diff {}'.format( diff ))

        w_best , w_first = w[ 'best' ] , w[ 'first' ]
        total_reward = 0.0
        l = 0

        self.T = 4
        state = augment_state( state , state )

        for i in range( 1000 ):
            i *= 0.01

            if (i > 4.6):
                i -= 4.2
                self.T = 2
                action = self.get_action( w_best , i )

            elif (i > 2):
                i -= 2
                self.T = 2
                action = self.get_action( w_best , i )
            else:
                action = self.get_action( w_first , i )

            next_state , reward , terminal , _ = env.step( action )
            action = np.reshape( action , (1 , -1) )
            l += self.teach( state , action )
            # sample a state every 2
            next_state = augment_state(state, next_state)

            if i % 2 == 0:
                self.memory.collect(state, action, reward, next_state, terminal)
            total_reward += reward
            state = next_state

            if terminal:
                break
        return l , total_reward, i



def augment_state(s , s1):

    s = np.reshape( np.array( s ) , (1 , -1) )
    s1 = np.reshape( np.array( s1 ) , (1 , -1) )

    idxs = [ 22 , 24 , 26 , 28 , 30 ]

    vel = (s1[ : , idxs ] - s[ : , idxs ]) / (0.01)
    # keep information of the environment like difficulty
    return np.reshape( np.append( s1[:,:38] , np.append(vel, s1[:,38:])) , (1 , -1) )