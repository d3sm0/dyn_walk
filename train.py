import tensorflow as tf

from ou_noise import OUNoise

tf.logging.set_verbosity(tf.logging.INFO)


class Worker(object):
    def __init__(self, env, agent, max_steps=None, batch_size=64, gamma=0.99):
        self.agent = agent
        self.env = env
        # self.ou = OUNoise ( action_dimension=agent.act_space )
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.gamma = gamma

    def run(self, sess, coord):
        t = 0
        with sess.as_default(), sess.graph.as_default():

            summarize = False
            self.agent.sync()
            try:
                while not coord.should_stop():
                    t += 1
                    self.state = self.env.reset()
                    # self.noise = self.ou.reset()

                    timesteps, tot_rw = self.sample(summarize=summarize)
                    summarize = False

                    if t % 100 == 0:
                        print('Agent {} downloading params. Next th {}'.format(self.agent.name, th))
                        self.agent.sync()

                    if self.agent.name == 'worker_0' and t % 5 == 0:
                        summarize = True

                        tf.logging.info(
                            'Master ep  {}, latest ep reward {}, of steps {}'.format(t, tot_rw, timesteps))
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
