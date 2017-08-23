import tensorflow as tf

from ou_noise import OUNoise

tf.logging.set_verbosity ( tf.logging.INFO )

class Worker ( object ):
    def __init__(self , env , agent , max_steps=None, batch_size = 64, gamma = 0.99):
        self.agent = agent
        self.env = env

        # self.ou = OUNoise ( action_dimension=agent.act_space )

        self.t = 0
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.gamma = gamma

    def run(self , sess , coord):


        with sess.as_default () , sess.graph.as_default ():

            last_t = 0
            summarize = False


            try:
                while not coord.should_stop ():


                    self.state = self.env.reset ()
                    # self.noise = self.ou.reset()
                    self.agent.sync ()

                    timesteps,tot_rw= self.sample (summarize=summarize)
                    summarize = False

                    self.t += 1

                    if self.agent.name == 'worker_0' and self.t - last_t >5:
                        last_t = self.t
                        summarize = True

                        tf.logging.info (
                            'Master ep  {}, latest ep reward {}, of steps {}'.format ( self.t , tot_rw , timesteps ) )

                    if self.max_steps is not None and self.t > self.max_steps:
                        tf.logging.info ( 'Hopefully i learnt something...test me...' )
                        coord.should_stop ()

            except tf.errors.CancelledError:
                return

    def sample(self, summarize = False):

        terminal = False

        t , tot_rw = 0 , 0
        # f_in = self.agent.reset()

        while not terminal:
            # if stochastic remove exploration noise

            action = self.agent.get_action ( self.state )
            # action = action + self.ou.noise()

            next_state , reward , terminal , _ = self.env.step ( action)
            self.agent.memory.collect ( self.state , action , reward , next_state , terminal )
            self.agent.think (batch_size=self.batch_size, gamma = self.gamma, summarize  = summarize)
            self.state = next_state
            t += 1
            tot_rw += reward

            if terminal:
                break

        if summarize:

            ep_summary = tf.Summary ()

            ep_summary.value.add ( simple_value=tot_rw , tag='eval/total_rw' )
            ep_summary.value.add ( simple_value=t, tag='eval/ep_length' )

            self.agent.writer.add_summary ( ep_summary , self.t )
            self.agent.writer.flush ()

        return t , tot_rw