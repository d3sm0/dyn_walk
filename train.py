import tensorflow as tf
import gym

from ou_noise import OUNoise

tf.logging.set_verbosity(tf.logging.INFO)


class Worker(object):
    def __init__(self, env_name, agent, max_steps=None, batch_size=64, gamma=0.99):
        self.agent = agent
        self.env_name = env_name
        self.env = None
        # self.ou = OUNoise ( action_dimension=agent.act_space )
        self.t = 0
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.gamma = gamma

    def run(self, sess, stop_training_event):
        self.agent.init()
        self.env = gym.make(self.env_name)
        last_t = 0
        summarize = False
        with sess.as_default(), sess.graph.as_default():
            try:
                while not stop_training_event.is_set():
                    self.agent.sync()
                    self.state = self.env.reset()
                    # self.noise = self.ou.reset()

                    timesteps, tot_rw = self.sample(summarize=summarize)
                    summarize = False
                    self.t += 1

                    if self.agent.name == 'worker_0' and self.t - last_t > 5:
                        last_t = self.t
                        summarize = True
                        tf.logging.info(
                            'Master ep  {}, latest ep reward {}, of steps {}'.format(self.t, tot_rw, timesteps))

                    if self.max_steps is not None and self.t > self.max_steps:
                        tf.logging.info('Hopefully i learnt something...test me...')
                        stop_training_event.set()
            except tf.errors.CancelledError:
                return

    def sample(self, summarize=False):

        terminal = False

        t, tot_rw = 0, 0
        while not terminal:
            # if stochastic remove exploration noise
            action = self.agent.get_action(self.state)  # + self.ou.noise()
            next_state, reward, terminal, _ = self.env.step(action)
            self.agent.memory.collect(self.state, action, reward, next_state, terminal)
            self.agent.think(batch_size=self.batch_size, gamma=self.gamma, summarize=summarize)
            self.state = next_state
            t += 1
            tot_rw += reward

        if summarize:
            ep_summary = tf.Summary()

            ep_summary.value.add(simple_value=tot_rw, tag='eval/total_rw')
            ep_summary.value.add(simple_value=t, tag='eval/ep_length')

            self.agent.writer.add_summary(ep_summary, self.t)
            self.agent.writer.flush()

        return t, tot_rw
