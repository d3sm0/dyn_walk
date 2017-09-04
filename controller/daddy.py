import numpy as np
import pickle
import tensorflow as tf
from memory.rm import Experience

file_name = 'controller/w.pkl'


class Daddy( object ):
    def __init__(self , target , env_dim , buffer_size=1e6, batch_size = 32):
        self.target = target
        self.action = tf.placeholder( 'float32' , shape=[ None , env_dim[ 1 ] ] , name='daddy_act' )
        self.loss = tf.reduce_mean( tf.squared_difference( self.action , target.actor.mu_hat ) )
        self.train_step = tf.train.AdamOptimizer().minimize( self.loss )
        # transfer update to critic

        # syncing only feature extraction
        actor_params = [ v for v in target.actor.params if 'policy' not in v.name ]
        self.sync_op = target.update_target( actor_params , target.critic.params[ :len(actor_params) ] ,
                                          tau=1 )
        self.T = 4
        self.w = self.load_w( file_name=file_name )
        self.memory = Experience(buffer_size=buffer_size, batch_size=batch_size)

    def teach(self , state , action):
        sess = tf.get_default_session()

        state, action, _ ,_, _ = self.memory.select()

        l , _ = sess.run( [ self.loss , self.train_step ] ,
                          feed_dict={self.target.actor.state: state , self.action: action} )
        return l

    def sync(self):
        sess = tf.get_default_session()
        sess.run( self.sync_op )

    def dump_model(self , file_name , w):
        with open( file_name , 'wb' ) as f:
            pickle.dump( w , f )
            print('Model dumped')

    def output(self , a , t):
        y = 0

        for i in range( 4 ):
            y += a[ i ] * np.sin( (i + 1) * np.pi * 2 * t / self.T + a[ i + 4 ] )
        return y

    def get_policy(self , w , t):

        policy = [ -self.output( w[ 0 ] , t ) , self.output( w[ 1 ] , t ) , -self.output( w[ 2 ] , t ) ,
                   self.output( w[ 3 ] , t ) , self.output( w[ 4 ] , t ) , self.output( w[ 5 ] , t ) ,
                   -self.output( w[ 6 ] , t ) , -self.output( w[ 7 ] , t ) , self.output( w[ 8 ] , t ) ,
                   self.output( w[ 0 ] , t + self.T / 2 ) , -self.output( w[ 1 ] , t + self.T / 2 ) ,
                   self.output( w[ 2 ] , t + self.T / 2 ) ,
                   -self.output( w[ 3 ] , t + self.T / 2 ) , -self.output( w[ 4 ] , t + self.T / 2 ) ,
                   -self.output( w[ 5 ] , t + self.T / 2 ) ,
                   self.output( w[ 6 ] , t + self.T / 2 ) , self.output( w[ 7 ] , t + self.T / 2 ) ,
                   -self.output( w[ 8 ] , t + self.T / 2 ) , ]

        return policy

    def load_w(self , file_name):
        # try:
        with open( file_name , 'rb' ) as f:
            w = pickle.load( f )
        return w
        # except IOError:
        #     print('File not found')

    def save_memory(self , file_name):
        try:
            with open( file_name , 'wb' ) as f:
                pickle.dump( self.memory.buffer , f )
                tf.logging.info( 'Saved memory' )
        except Exception as e:
            tf.logging.info( e )

    def load_memory(self , file_name):
        try:
            with open( file_name , 'rb' ) as f:
                self.memory.buffer = np.copy(pickle.load(f))
                tf.logging.info( 'Load daddy memory' )
        except Exception as e:
            tf.logging.info( e )
            
    def get_action(self, t):

        # selected number
        i = t * 0.01

        if (i > 4.6):
            i -= 4.2
            self.T = 2
            action = self.get_policy( self.w[ 'best' ] , i )
        elif (i > 2):
            i -= 2
            self.T = 2
            action = self.get_policy( self.w[ 'best' ] , i )
        else:
            action = self.get_policy( self.w[ 'first' ] , i )
        return action

    # def sample(self , env, diff = 0):
    # 
    #     state = env.reset( difficulty=diff)
    # 
    #     w_best , w_first = self.w[ 'best' ] , self.w[ 'first' ]
    #     total_reward = 0.0
    #     l = 0
    # 
    #     self.T = 4
    # 
    #     for i in range( 1000 ):
    #         i *= 0.01
    # 
    #         if (i > 4.6):
    #             i -= 4.2
    #             self.T = 2
    #             action = self.get_action( w_best , i )
    # 
    #         elif (i > 2):
    #             i -= 2
    #             self.T = 2
    #             action = self.get_action( w_best , i )
    #         else:
    #             action = self.get_action( w_first , i )
    # 
    #         next_state , reward , terminal , _ = env.step( action )
    #         self.memory.add( state , action , reward , next_state , terminal )
    # 
    #         l += self.teach( state , action )
    # 
    #         total_reward += reward
    #         state = next_state
    # 
    #         if terminal:
    #             break
    #     return l , total_reward/i , i


    def experience(self, epsiodes, env, writer):

        # load pre trained model
        losses = []

        # Not use surrogate reward
        env.augment_rw = False

        for ep in range(epsiodes):

            l, tot_rw, timesteps = self.sample(env= env)
            losses.append(l)

            if ep % 10 == 0:
                ep_summary = tf.Summary()

                ep_summary.value.add( simple_value=tot_rw/timesteps , tag='daddy/av_rw' )
                ep_summary.value.add( simple_value=timesteps , tag='daddy/timesteps' )
                ep_summary.value.add( simple_value= np.mean(l), tag='daddy/avg_l' )

                writer.add_summary( ep_summary , ep)
                writer.flush()

                tf.logging.info(
                    'Master ep  {}, latest ep reward {}, of steps {}'.format( ep , tot_rw , timesteps ) )


        # transfer learning
        self.sync()
        # save memory
        self.save_memory('memory.pkl')
        # transfer memory
        # agent.memory.buffer = copy.deepcopy(daddy.memory.buffer)




