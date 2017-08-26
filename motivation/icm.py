import tensorflow as tf
from tensorflow.contrib.layers import fully_connected , flatten , summarize_activation , summarize_tensor
import numpy as np


def _shared_model(phi , h_size,act=tf.nn.elu):
    h1 = act( _linear( phi , h_size , 'h0' , _normalized_columns_initializer( 0.01 ) ), name = 'h1_shared')
    # h1 = act( _linear( h1 , h_size , 'h1' , _normalized_columns_initializer( 0.01 ) ) )
    # summarize_activation( h0 )
    summarize_activation( h1 )
    return h1


def _normalized_columns_initializer(std=1.0):
    def _initializer(shape , dtype=None , partition_info=None):
        out = np.random.randn( *shape ).astype( np.float32 )
        out *= std / np.sqrt( np.square( out ).sum( axis=0 , keepdims=True ) )
        return tf.constant( out )

    return _initializer


def _linear(x , size , name , initializer=None , bias_init=0):
    w = tf.get_variable( name + "/w" , [ x.get_shape()[ 1 ] , size ] , initializer=initializer )
    b = tf.get_variable( name + "/b" , [ size ] , initializer=tf.constant_initializer( bias_init ) )
    return tf.matmul( x , w ) + b


class ICM( object ):
    def __init__(self , env_dims , h_size = 256, alpha=50 , beta=0.2 , lr=1e-4 , batch_size=32 , act=tf.nn.elu):
        obs_space , act_space , bound = env_dims
        len_features = h_size * act_space

        self.state = tf.placeholder( 'float32' , shape=[ None , obs_space ] , name='state' )
        self.next_state = tf.placeholder( 'float32' , shape=[ None , obs_space ] , name='next_state' )
        self.action = tf.placeholder( 'float32' , shape=[ None , act_space ] , name='action' )

        # notice that we share the hidden layer
        #         with tf.variable_scope('predictor'):
        phi1 = _shared_model( self.state, h_size)
        with tf.variable_scope( tf.get_variable_scope() , reuse=True ):
            phi2 = _shared_model( self.next_state, h_size)
        # inverse model

        g = tf.concat( values=[ phi1 , phi2 ] , axis=1 )
        g = act( _linear( g , h_size , 'f1' , _normalized_columns_initializer( 0.01 ) ), name = 'f1')

        mu_hat = tf.tanh( _linear( g , act_space , 'mu_hat' , _normalized_columns_initializer( 0.01 ) ), name = 'mu_hat')
        self.mu_hat = tf.clip_by_value( mu_hat , clip_value_min=bound[ 0 ] , clip_value_max=bound[ 1 ] )

        # forward model
        f = tf.concat( values=[ phi1 , self.action ] , axis=1 )
        f = act( _linear( f , h_size, 'g1' , tf.random_uniform_initializer( minval=-0.003 , maxval=0.003 ) ), name = 'g1')
        # predicted next state
        self.phi2_hat = act( _linear( f , h_size  , 'phi2_hat' ,
                                      tf.random_uniform_initializer( minval=-0.003 , maxval=0.003 ) ), name= 'phi2_hat')

        self.fwd_loss = len_features * 0.5 * tf.reduce_mean( tf.squared_difference( self.phi2_hat , phi2 ) ,
                                                             name='forward_loss' )
        self.inv_loss = 0.5 * tf.reduce_mean( tf.squared_difference( self.mu_hat , self.action ) , name='inverse_loss' )

        self.loss = alpha * tf.add(self.inv_loss * (1 - beta),self.fwd_loss * beta, name='loss')

        self.params = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES )

        grads = tf.gradients( self.loss * batch_size , self.params )

        # grads,_ =tf.clip_by_global_norm(grads, 40)
        self.train_step = tf.train.AdamOptimizer( learning_rate=lr).apply_gradients( zip( grads , self.params ), global_step=tf.contrib.framework.get_global_step())

        summarize_activation( g )
        summarize_activation( self.mu_hat )
        summarize_activation( f )
        summarize_activation( self.phi2_hat )

        # scalar = [ self.fwd_loss , self.inv_loss ,
        #            self.loss ]
        #
        # self.summary_ops = build_summaries( scalar=scalar , hist = None)

    def get_action(self , state , next_state):
        sess = tf.get_default_session()
        return sess.run( self.mu_hat , feed_dict={self.state: state , self.next_state: next_state} )

    def get_state(self , state , action):
        sess = tf.get_default_session()
        return sess.run( self.phi2_hat , feed_dict={self.state: state , self.action: action} )

    def get_bonus(self , state , next_state , action , rwd_scale=0.5):
        sess = tf.get_default_session()

        state = np.reshape(state, (1,-1))
        next_state = np.reshape(next_state,(1,-1))
        action = np.reshape(action,(1,-1))

        loss = sess.run( self.fwd_loss ,
                         feed_dict={self.state:  state  , self.next_state:  next_state  , self.action:  action} )
        bonus_rwd = loss * rwd_scale

        return bonus_rwd

    def train(self , state , next_state , action):
        sess = tf.get_default_session()

        feed_dict = {self.state: state , self.next_state: next_state , self.action: action}
        loss = sess.run( [ self.fwd_loss , self.inv_loss , self.loss ] , feed_dict=feed_dict )

        return loss



