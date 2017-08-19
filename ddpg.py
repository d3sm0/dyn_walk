import tensorflow as tf
from tensorflow.contrib.layers import fully_connected , summarize_activation , convolution2d , batch_norm , flatten , \
    summarize_tensor


# if determistic policyr use relu activation
# relu activation should approximate hirerarchial rbf kernel behavior

class Actor ( object ):
    def __init__(self , obs_space , action_space , action_bound , h_size , lr=1e-4, act = tf.nn.elu, stochastic = False):
        self.state = tf.placeholder ( 'float32' , shape=[ None , obs_space ] , name='state' )
        self.grads = tf.placeholder ( 'float32' , shape=[ None , action_space ] , name='gradients' )

        self.mu_hat = self.build_network ( h_size , action_space , action_bound, act, stochastic=stochastic)

        self.params = tf.get_collection ( tf.GraphKeys.GLOBAL_VARIABLES , scope=tf.get_variable_scope ().name )

        grads = tf.gradients ( self.mu_hat , self.params , -self.grads )

        self.train = tf.train.AdamOptimizer ( learning_rate=lr ).apply_gradients ( zip ( grads , self.params ) ,
                                                                                   global_step=tf.contrib.framework.get_global_step () )

    def build_network(self , h_size , action_space , action_bound, act, stochastic):

        h1 = fully_connected ( inputs=self.state , num_outputs=h_size[0] , activation_fn=act)
        h2 = fully_connected ( inputs=h1 , num_outputs=h_size [1], activation_fn=act)

        mu_hat = fully_connected ( inputs=h2 , num_outputs=action_space , activation_fn=tf.nn.tanh ,
                                   weights_initializer=tf.random_uniform_initializer ( minval=-0.003 , maxval=0.003 ) )

        # this should be initialized by a gamma distribution

        if stochastic:
            sigma = tf.get_variable ( name='sigma' , shape=(action_space ,) , dtype=tf.float32 ,
                                                   initializer=tf.random_uniform_initializer ( minval=0 , maxval=1 ))
            sigma = tf.reshape ( sigma , (-1 , action_space) )

            mu_hat = mu_hat + tf.random_normal ( shape=tf.shape ( mu_hat ) ) * sigma

            summarize_tensor ( sigma , tag='sigma' )

        scaled = tf.multiply ( mu_hat , action_bound )

        summarize_activation ( h1 )
        summarize_activation ( h2 )
        summarize_activation ( mu_hat)

        return scaled


class Critic ( object ):
    def __init__(self , obs_space , action_space , h_size , lr=1e-3, act = tf.nn.elu):
        self.state = tf.placeholder ( 'float32' , shape=[ None , obs_space ] , name='state' )
        self.action = tf.placeholder ( 'float32' , shape=[ None , action_space ] , name='action' )

        self.q = tf.placeholder ( 'float32' , shape=[ None , 1 ] , name='q' )

        self.q_hat = self.build_network ( h_size , action_space, act)

        self.params = tf.get_collection ( tf.GraphKeys.GLOBAL_VARIABLES , scope=tf.get_variable_scope ().name )

        self.critic_loss = tf.reduce_mean ( tf.squared_difference ( self.q , self.q_hat ) , name='critic_loss' )

        self.train = tf.train.AdamOptimizer ( learning_rate=lr ).minimize ( self.critic_loss ,
                                                                            global_step=tf.contrib.framework.get_global_step () )

        self.action_grads = tf.gradients ( self.q_hat , self.action )

    def build_network(self , h_size , action_space, act):
        h0 = fully_connected ( inputs=self.state , num_outputs=h_size[0] , activation_fn=act )

        h1 = fully_connected ( inputs=h0 , num_outputs=h_size[1] , activation_fn=act)

        w1 = tf.get_variable ( 'w1' , shape=[ h_size[1] , h_size[1] ] , dtype=tf.float32 )
        w2 = tf.get_variable ( 'w2' , shape=[ action_space , h_size[1] ] , dtype=tf.float32 )
        b2 = tf.get_variable ( 'b2' , shape=[ h_size[1] ] , dtype=tf.float32 )

        h2 = tf.nn.elu ( tf.matmul ( h1 , w1 ) + tf.matmul ( self.action , w2 ) + b2 )

        q_hat = fully_connected ( inputs=h2 , num_outputs=1 , activation_fn=tf.identity ,
                                  weights_initializer=tf.random_uniform_initializer ( minval=-0.003 , maxval=0.003 ) )

        summarize_activation ( h0 )
        summarize_activation ( h1 )
        summarize_activation ( h2 )
        summarize_activation ( q_hat )

        return q_hat


def build_summaries(scalar=None , hist=None):
    for v in scalar:
        tf.summary.scalar ( v.name.replace ( ':' , "_" ) + 'mean' , tf.reduce_mean ( v ) )

    for h in hist:
        tf.summary.histogram ( h.name.replace ( ':' , "_" ) , h )

    summary_ops = tf.get_collection ( tf.GraphKeys.SUMMARIES , scope=tf.get_variable_scope ().name )
    summary_ops = [ s for s in summary_ops if 'target' not in s.name ]

    return tf.summary.merge ( summary_ops )