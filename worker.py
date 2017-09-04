from agent import Agent
import tensorflow as tf
from utils.env_wrapper import  EnvWrapper
from memory.rm import Experience

class Worker( object ):
    def __init__(self , target , config , log_dir):
        self.env , env_dim, _  = self.init_environment( config )
        self.agent = Agent( name='local' ,target=target , split_obs=None , log_dir=log_dir )

        self.agent.initialize(env_dims=env_dim , h_size=config[ 'H_SIZE' ] ,
                            policy=config[ 'POLICY' ] , act=eval(config[ 'ACTIVATION' ] ), split_obs=None )
        self.writer = tf.summary.FileWriter( self.agent.log_dir, filename_suffix="episode_metrics" )
        self.ep_summary = tf.Summary()
        self.memory = Experience(buffer_size=self.agent.memory.buffer_size,batch_size=self.agent.memory.batch_size, log_dir=log_dir)


    def warmup(self, ep = 5):
        for _ in range(ep):
            self.env.reset()
            terminal = False
            while not terminal:
                action = self.env.env.action_space.sample()
                _,_,terminal,_ = self.env.step(action)
        tf.logging.info('Warm up ended')

    def unroll(self, curr_ep = 0):
        state = self.env.reset()
        self.agent.reset()

        terminal = False
        timesteps , tot_rw , tot_q , tot_td = 0 , 0 , 0 , 0
        while not terminal:
            # if stochastic remove exploration noise
            action = self.agent.get_action( state ).flatten() + self.agent.ou.noise()
            next_state , reward , terminal , _ = self.env.step( action )
            step = (state , action , reward , next_state , terminal)
            self.memory.collect(step,curr_ep)
            td , q = self.agent.process( *step )

            state = next_state

            timesteps += 1
            tot_td += td
            tot_q += q
            tot_rw += reward
        return tot_td , tot_q , tot_rw , timesteps

    def report_metrics(self , tot_td , tot_q , tot_rw , timesteps , episode_count):
        self.ep_summary.value.add( simple_value=tot_rw / timesteps , tag='eval/avg_rw ')
        self.ep_summary.value.add( simple_value=tot_q / timesteps , tag='eval/avg_q' )
        self.ep_summary.value.add( simple_value=tot_td / timesteps , tag='eval/avg_td' )
        self.ep_summary.value.add( simple_value=tot_rw , tag='eval/r' )
        self.ep_summary.value.add( simple_value=tot_td , tag='eval/total_td' )
        self.ep_summary.value.add( simple_value=tot_q , tag='eval/total_q' )
        self.ep_summary.value.add( simple_value=timesteps , tag='eval/ep_length' )
        # try:
        #     self.ep_summary.value.add( simple_value=self.env.r / timesteps , tag='eval/avg_rw' )
        #     self.ep_summary.value.add( simple_value=self.env.r , tag='eval/total_rw' )
        # except AttributeError:
        #     pass
        self.writer.add_summary( self.ep_summary , episode_count)
        self.writer.flush()

    @staticmethod
    def init_environment(config):
        env = None
        env_dims = None
        split_obs = None
        if config[ 'ENV_NAME' ] == 'osim':
            from osim.env import RunEnv
            try:
                env = EnvWrapper( RunEnv , visualize=False , augment_rw=config[ 'USE_RW' ] ,
                                  concat=config[ 'CONCATENATE_FRAMES' ] , add_time=False,
                                  normalize=config[ 'NORMALIZE' ] , add_acceleration=7 )
                env_dims = env.get_dims()
                split_obs = env.split_obs
            except Exception as e:
                tf.logging.info( 'Environment not found' )
                raise e
        else:
            try:
                import gym
                env = gym.make( config[ 'ENV_NAME' ] )
                env_dims = (
                    env.observation_space.shape[ 0 ] , env.action_space.shape[ 0 ] ,
                    (env.action_space.low , env.action_space.high))
            except:
                raise NotImplementedError()
        return env , env_dims, split_obs
