import tensorflow as tf
from tensorflow.contrib.layers import fully_connected , summarize_activation , summarize_tensor

class Actor( object ):
    def __init__(self , obs_space , action_space , action_bound , h_size , lr=1e-4 , act=tf.nn.relu , policy='det' ,
                 split_obs=None):
        self.state = tf.placeholder( 'float32' , shape=[ None , obs_space ] , name='state' )

        if split_obs is not None:
            self.internal_state = self.state[ : , :split_obs ]
            self.external_state = self.state[ : , split_obs: ]

        self.grads = tf.placeholder( 'float32' , shape=[ None , action_space ] , name='gradients' )

        self.mu_hat = self.build_network( h_size , action_space , action_bound , act , policy ,
                                          has_external=bool( split_obs ) )

        self.params = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , scope=tf.get_variable_scope().name )

        self.actor_grads = tf.gradients( self.mu_hat , self.params , -self.grads )

        self.train = tf.train.AdamOptimizer( learning_rate=lr ).apply_gradients( zip( self.actor_grads , self.params ) ,
                                                                                 global_step=tf.contrib.framework.get_global_step() )

    def build_network(self , h_size , action_space , action_bound , act , policy , has_external=True):

        if has_external:

            p0 = fully_connected( inputs=self.internal_state , num_outputs=h_size[ 0 ] , activation_fn=act )

            p1 = fully_connected( inputs=self.external_state , num_outputs=h_size[ 0 ] , activation_fn=act )

            p1 = fully_connected( inputs=p1 , num_outputs=h_size[ 0 ] , activation_fn=act )
            h1 = fully_connected( inputs=p0 , num_outputs=h_size[ 1 ] , activation_fn=act )
            h2 = fully_connected( inputs=h1 , num_outputs=h_size[ 1 ] , activation_fn=act )

            h2 = tf.concat( (h2 , p1) , axis=1 )

            summarize_activation( p0 )
            summarize_activation( p1 )

        else:
            h1 = fully_connected( inputs=self.state , num_outputs=h_size[ 0 ] , activation_fn=act )
            h2 = fully_connected( inputs=h1 , num_outputs=h_size[ 1 ] , activation_fn=act )

        mu_hat = fully_connected( inputs=h2 , num_outputs=action_space , activation_fn=tf.nn.tanh ,
                                  weights_initializer=tf.random_uniform_initializer( minval=-0.003 , maxval=0.003 ) ,
                                  scope='policy' )

        if policy == 'stochastic':
            mu_hat = self.stochastic_policy( mu_hat , action_space )

        elif policy == 'sin':
            mu_hat = self.sin_policy( h2 , mu_hat , action_space )

        action = tf.clip_by_value( mu_hat , 0 , 1 , name='scaled' )

        summarize_activation( h1 )
        summarize_activation( h2 )

        summarize_activation( mu_hat )

        return action

    def sin_policy(self , h2 , mu_hat , action_space):

        amplitude = fully_connected( inputs=h2 , num_outputs=action_space , activation_fn=None )

        mu_hat = tf.map_fn( fn=lambda x: tf.sin( x * 2 ) + 1 / 2 , elems=mu_hat )
        mu_hat = tf.multiply( amplitude , mu_hat )

        return mu_hat

    def stochastic_policy(self , mu_hat , action_space):

        # var = fully_connected( inputs=h2 , num_outputs=action_space , activation_fn=tf.nn.softplus )

        var = tf.nn.softplus( tf.get_variable( 'sd' , dtype=tf.float32 , shape=(action_space ,) ) )
        var = tf.reshape( var , (-1 , action_space) )

        action = mu_hat + tf.random_normal( shape=tf.shape( mu_hat ) ) * tf.sqrt( var )
        summarize_tensor( var , tag='var' )
        summarize_tensor( action , tag='sampled_action' )


class Critic( object ):
    def __init__(self , obs_space , action_space , h_size , lr=1e-3 , act=tf.nn.relu , split_obs=None):
        self.state = tf.placeholder( 'float32' , shape=[ None , obs_space ] , name='state' )
        self.action = tf.placeholder( 'float32' , shape=[ None , action_space ] , name='action' )

        self.q = tf.placeholder( 'float32' , shape=[ None , 1 ] , name='q' )

        if split_obs is not None:
            self.internal_state = self.state[ : , :split_obs ]
            self.external_state = self.state[ : , split_obs: ]

        self.q_hat = self.build_network( h_size , action_space , act , has_external=bool( split_obs ) )

        self.params = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , scope=tf.get_variable_scope().name )

        self.critic_loss = tf.reduce_mean( tf.squared_difference( self.q , self.q_hat ) , name='critic_loss' )

        self.train = tf.train.AdamOptimizer( learning_rate=lr ).minimize( self.critic_loss ,
                                                                          global_step=tf.contrib.framework.get_global_step() )

        self.action_grads = tf.gradients( self.q_hat , self.action )

    def build_network(self , h_size , action_space , act , has_external):

        if has_external:

            p0 = fully_connected( inputs=self.internal_state , num_outputs=h_size[ 0 ] , activation_fn=act )
            p1 = fully_connected( inputs=self.external_state , num_outputs=h_size[ 0 ] , activation_fn=act )

            p1 = fully_connected( inputs=p1 , num_outputs=h_size[ 0 ] , activation_fn=act )

            h1 = fully_connected( inputs=p0 , num_outputs=h_size[ 0 ] , activation_fn=act )

            h1 = tf.concat( (h1 , p1) , axis=1 )

            summarize_activation( p0 )
            summarize_activation( p1 )

        else:
            h1 = fully_connected( inputs=self.state , num_outputs=h_size[ 0 ] , activation_fn=act )

        w1 = tf.get_variable( 'w1' , shape=[ h_size[ 0 ] , h_size[ 1 ] ] , dtype=tf.float32 )
        w2 = tf.get_variable( 'w2' , shape=[ action_space , h_size[ 1 ] ] , dtype=tf.float32 )
        b2 = tf.get_variable( 'b2' , shape=[ h_size[ 1 ] ] , dtype=tf.float32 )

        h2 = act( tf.matmul( h1 , w1 ) + tf.matmul( self.action , w2 ) + b2 )

        q_hat = fully_connected( inputs=h2 , num_outputs=1 , activation_fn=None ,
                                 weights_initializer=tf.random_uniform_initializer( minval=-0.003 , maxval=0.003 ) ,
                                 scope='value' )

        summarize_activation( h2 )
        summarize_activation( q_hat )

        return q_hat


def build_summaries(scalar=None , hist=None):
    for v in scalar:
        tf.summary.scalar( v.name.replace( ':' , "_" ) + 'mean' , tf.reduce_mean( v ) )

    for h in hist:
        tf.summary.histogram( h.name.replace( ':' , "_" ) , h )

    summary_ops = tf.get_collection( tf.GraphKeys.SUMMARIES , scope=tf.get_variable_scope().name )
    summary_ops = [ s for s in summary_ops if 'target' not in s.name ]

    return tf.summary.merge( summary_ops )


def lrelu(x , alpha=0.05):
    return tf.nn.relu( x ) - alpha * tf.nn.relu( -x )
