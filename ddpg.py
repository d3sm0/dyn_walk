import tensorflow as tf
from tensorflow.contrib.layers import fully_connected , summarize_activation , summarize_tensor


def lrelu(x , alpha=0.2 , name=None):
    return tf.subtract(tf.nn.relu(x) , alpha * tf.nn.relu(-x) , name=name)


class DDPG(object):
    def __init__(self , obs_space , action_space , action_bound , h_size , act=tf.nn.elu , policy='det' ,
                 split_obs=None , lr=1e-3):

        self.state = tf.placeholder('float32' , shape=[None , obs_space] , name='state')
        self.action = tf.placeholder('float32' , shape=[None , action_space] , name='action')

        self.q = tf.placeholder('float32' , shape=[None , 1] , name='q')

        h1 = self.shared_network(act=act , h_size=h_size)

        self.mu_hat = self.policy_network(h1 , bound=action_bound , action_space=action_space , act=act , policy=policy)
        self.q_hat = self.value_network(h1 , act=act)

        self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES , scope=tf.get_variable_scope().name)
        self.critic_loss = tf.reduce_mean(tf.squared_difference(self.q , self.q_hat) , name='critic_loss')

        self.train_critic = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.critic_loss ,
                                                                              global_step=tf.contrib.framework.get_global_step())
        self.action_grads = tf.gradients(self.q_hat , self.action)[0]
        self.actor_grads = tf.gradients(self.mu_hat , self.params , -self.action_grads)

        self.train_actor = tf.train.AdamOptimizer(learning_rate=lr).apply_gradients(
            zip(self.actor_grads , self.params) ,
            global_step=tf.contrib.framework.get_global_step())

    def shared_network(self , h_size=(256 , 128) , split_obs=None , act=lrelu):

        if split_obs:

            h_size = 64

            self.internal_state = self.state[: , :split_obs]
            self.external_state = self.state[: , split_obs:]

            p0 = fully_connected(inputs=self.internal_state , num_outputs=h_size , activation_fn=act)
            p0 = fully_connected(inputs=p0 , num_outputs=h_size , activation_fn=act)

            p1 = fully_connected(inputs=self.external_state , num_outputs=h_size , activation_fn=act)
            p1 = fully_connected(inputs=p1 , num_outputs=h_size , activation_fn=act)

            h1 = tf.concat((p0 , p1) , axis=1)
            summarize_activation(p0)
            summarize_activation(p1)

        else:
            h = fully_connected(inputs=self.state , num_outputs=h_size[0] , activation_fn=act)
            for size in h_size:
                h = fully_connected(inputs=h , num_outputs=size , activation_fn=act)
            summarize_activation(h)

        return h

    def policy_network(self , h1 , bound , action_space , h_size=64 , policy='det' , act=lrelu):

        h2 = fully_connected(inputs=h1 , num_outputs=h_size , activation_fn=act)

        if policy == 'stochastic':
            mu_hat = self.stochastic_policy(h2 , action_space)
        else:
            mu_hat = fully_connected(inputs=h2 , num_outputs=action_space , activation_fn=tf.nn.tanh ,
                                     weights_initializer=tf.random_uniform_initializer(minval=-0.003 , maxval=0.003) ,
                                     scope='policy')

        mu_hat = tf.clip_by_value(mu_hat , clip_value_min=bound[0] , clip_value_max=bound[1])

        summarize_activation(h2)
        summarize_activation(mu_hat)
        return mu_hat

    def value_network(self , h1 , h_size=64 , act=lrelu):
        #
        # w1 = tf.get_variable( 'w1' , shape=[ h_size[ 0 ] , h_size[ 1 ] ] , dtype=tf.float32 )
        # w2 = tf.get_variable( 'w2' , shape=[ action_space , h_size[ 1 ] ] , dtype=tf.float32 )
        # b2 = tf.get_variable( 'b2' , shape=[ h_size[ 1 ] ] , dtype=tf.float32 )
        #
        # h2 = act( tf.matmul( h1 , w1 ) + tf.matmul( self.action , w2 ) + b2 )

        # concatenating (s,a)
        h2 = fully_connected(inputs=tf.concat((h1 , self.action) , axis=1) , num_outputs=h_size , activation_fn=act)

        q_hat = fully_connected(inputs=h2 , num_outputs=1 , activation_fn=None ,
                                weights_initializer=tf.random_uniform_initializer(minval=-0.003 , maxval=0.003) ,
                                scope='value')

        summarize_activation(h2)
        summarize_activation(q_hat)

        return q_hat

    def stochastic_policy(self , h2 , action_space):

        mu = fully_connected(inputs=h2 , num_outputs=action_space)
        logsd = tf.get_variable('sd' , dtype=tf.float32 , shape=(1 , action_space) , initializer=tf.zeros_initializer)
        #
        # var = tf.nn.softplus( tf.get_variable( 'sd' , dtype=tf.float32 , shape=(action_space ,) ) )
        # var = tf.reshape( var , (-1 , action_space) )

        action = mu + tf.random_normal(shape=tf.shape(mu)) * tf.exp(logsd)
        summarize_tensor(tf.exp(logsd) , tag='sd')
        summarize_tensor(mu , tag='mu')
        return action


def build_summaries(scalar=None , hist=None):
    for v in scalar:
        tf.summary.scalar(v.name.replace(':' , "_") + 'mean' , tf.reduce_mean(v))

    if hist is not None:
        for h in hist:
            tf.summary.histogram(h.name.replace(':' , "_") , h)

    summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES , scope=tf.get_variable_scope().name)
    summary_ops = [s for s in summary_ops if 'target' not in s.name]

    return tf.summary.merge(summary_ops)
