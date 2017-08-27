import tensorflow as tf
from ddpg import Actor , Critic , build_summaries
import numpy as np
from memory.pmr import Experience


class Agent( object ):
    def __init__(self , name , env_dims , target=None , writer=None , h_size=128 , batch_size=32 , memory_size=10e6 ,
                 policy='det' ,
                 act=tf.nn.elu , split_obs=None , motivation=None):

        self.obs_space , self.act_space , bound = env_dims
        with tf.variable_scope( name ):
            self.actor = Actor( self.obs_space , self.act_space , bound , h_size=h_size , policy=policy , act=act ,
                                split_obs=split_obs )
            self.critic = Critic( self.obs_space , self.act_space , h_size=h_size , act=act )

            if motivation is not None and name != 'target':
                self.motivation = motivation( env_dims=env_dims , act=act )

        # TODO test prioritized experience replay
        # self.memory = Experience ( batch_size=batch_size, buffer_size=int(memory_size))
        self.memory = Experience( batch_size=batch_size , memory_size=int( memory_size ) )

        if name != 'target':
            self.sync_op = [ self.update_target( self.actor.params , target.actor.params ) ,
                             self.update_target( self.critic.params , target.critic.params ) ]

            scalar = [ self.critic.q , self.critic.critic_loss ,
                       # self.motivation.fwd_loss , self.motivation.inv_loss ,
                       # self.motivation.loss
                       ]
            hist = [ self.critic.state , self.critic.action , self.actor.grads ]


            self.summary_ops = build_summaries( scalar=scalar , hist=hist )

            self.writer = writer

        self.name = name
        self.target = target
        self.gamma = 0.99

        tf.logging.info( 'Worker {} ready to go ...'.format( self.name ) )

    def summarize(self , feed_dict , global_step):

        sess = tf.get_default_session()

        summary = sess.run( self.summary_ops ,
                            feed_dict=feed_dict )

        self.writer.add_summary( summary , global_step=global_step )
        self.writer.flush()

    def sync(self):

        sess = tf.get_default_session()
        sess.run( self.sync_op )

    def train(self , state , action , q , next_state=None , get_summary=False):

        sess = tf.get_default_session()

        critic_loss , _ , global_step = sess.run(
            [ self.critic.critic_loss , self.critic.train , tf.contrib.framework.get_global_step() ] , feed_dict={
                self.critic.state: state ,
                self.critic.action: action ,
                self.critic.q: q
            } )

        # compute sample of the gradient

        sampled_action = self.get_action( state )
        sampled_grads = self.get_grads( state , sampled_action )
        #
        # fwd_grads = self.motivation.get_grads(state, next_state,action)
        #
        # sampled_grads = np.subtract(sampled_grads, fwd_grads)

        _ , gr = sess.run( [ self.actor.train, self.actor.actor_grads], feed_dict={
            self.actor.state: state ,
            self.actor.grads: sampled_grads
        } )


        if next_state is not None:
            icm_loss = self.motivation.train( state , next_state , action )


        if get_summary:
            feed_dict = {
                self.actor.state: state ,
                self.actor.grads: sampled_grads ,
                self.critic.state: state ,
                self.critic.action: action ,
                self.critic.q: q,
                # self.motivation.state:state,
                # self.motivation.next_state:next_state,
                # self.motivation.action:action
            }
            self.summarize( feed_dict , global_step )

            #
            # if self.motivation is not None:
            #     self.motivation.summarize(state, next_state, action, writer = self.writer)

        return critic_loss

    def get_grads(self , state , action):
        sess = tf.get_default_session()
        return sess.run( self.critic.action_grads , feed_dict={
            self.critic.state: state ,
            self.critic.action: action ,
        } )[ 0 ]

    def get_action(self , state):
        sess = tf.get_default_session()

        # TODO pre-process of state should not happen here
        if np.ndim( state ) != self.obs_space:
            state = np.reshape( state , (-1 , self.obs_space) )

        mu_hat = sess.run( self.actor.mu_hat ,
                           feed_dict={self.actor.state: state} )
        return mu_hat

    def get_q(self , state , action):

        sess = tf.get_default_session()
        if np.ndim( state ) != self.obs_space or np.ndim( action ) != self.act_space:
            state = np.reshape( state , (-1 , self.obs_space) )
            action = np.reshape( action , (-1 , self.act_space) )

        q_hat = sess.run( self.critic.q_hat , feed_dict={self.critic.state: state , self.critic.action: action} )

        # q_hat = np.clip(q_hat, (-40,40))
        return q_hat.ravel()

    # def think(self, summarize):
    #
    #     if self.memory.get_size() > self.memory.batch_size:
    #         s1_batch , a_batch , r_batch , s2_batch , t_batch = self.memory.select ( )
    #
    #         target_action = self.target.get_action ( s2_batch )
    #         target_q = self.target.get_q ( s2_batch , target_action )
    #
    #         y_i = [ ]
    #
    #         for k in range ( self.memory.batch_size ):
    #             if t_batch[ k ]:
    #                 y_i.append ( r_batch[ k ] )
    #             else:
    #                 y_i.append ( r_batch[ k ] + self.gamma * target_q[ k ] )
    #
    #         y_i = np.reshape ( y_i , (self.memory.batch_size , 1) )
    #
    #         c_loss = self.train ( s1_batch , a_batch , y_i , get_summary=summarize )



    def get_td(self , state , action , reward , next_state , terminal):

        target_action = self.target.get_action( next_state )

        target_q = self.target.get_q( next_state , target_action )[ 0 ]
        local_q = self.get_q( state , action )[ 0 ]
        q = reward

        if not terminal:
            q += self.gamma * target_q

        return np.abs( q - local_q ) , q

    def think(self , summarize):

        data , _ , idxs = self.memory.select()

        if data is not None:

            s1_batch , a_batch , r_batch , s2_batch , t_batch = self.memory.prepare_output( data )

            target_action = self.target.get_action( s2_batch )
            target_q = self.target.get_q( s2_batch , target_action )

            local_q = self.get_q( s1_batch , a_batch )
            # TODO check data structure here
            q = [ ]
            td = [ ]
            for k in range( self.memory.batch_size ):
                if t_batch[ k ]:
                    q.append( r_batch[ k ] )
                    td.append( r_batch[ k ] )
                else:
                    q.append( r_batch[ k ] + self.gamma * target_q[ k ] )
                    td.append( r_batch[ k ] + self.gamma * target_q[ k ] - local_q[ k ] )

            self.memory.priority_update( indices=idxs , priorities=np.abs( td ) )

            q = np.reshape( q , (self.memory.batch_size , 1) )

            self.train( s1_batch , a_batch , q , get_summary=summarize )

    @staticmethod
    # tau = 0.001
    def update_target(local , target , tau=0.01):
        params = [ ]
        for i in range( len( target ) ):
            params.append(
                target[ i ].assign( tf.multiply( local[ i ] , tau ) + tf.multiply( target[ i ] , 1. - tau ) ) )

        return params

    @staticmethod
    def global_trainer(grads , params):
        grads = [ grad for grad in grads if grad is not None ]
        grads , _ = tf.clip_by_global_norm( grads , 40 )
        train_op = tf.train.AdamOptimizer().apply_gradients( zip( grads , params ) ,
                                                             global_step=tf.contrib.framework.get_global_step() )
        return train_op
