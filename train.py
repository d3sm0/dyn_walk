import tensorflow as tf
import numpy as np
import multiprocessing
import gym

from ou_noise import OUNoise
from enum import Enum

tf.logging.set_verbosity(tf.logging.INFO)


class SimulationCommands(Enum):
    step = 0
    observe = 1
    reset = 2


class IsolatedEnv(object):
    def __init__(self, env_name):
        self.env_name = env_name
        pipe = multiprocessing.Pipe()
        self.pipe, pipe_process_end = pipe
        self.process = multiprocessing.Process(target=self.run_simulation, args=(env_name, pipe_process_end))
        self.process.start()

    @staticmethod
    def run_simulation(env_name, pipe):
        env = gym.make(env_name)
        while True:
            cmd, *args = pipe.recv()

            if cmd == SimulationCommands.step:
                retr = env.step(args)
                pipe.send(retr)
            elif cmd == SimulationCommands.observe:
                pass
            elif cmd == SimulationCommands.reset:
                retr = env.reset()
                pipe.send(retr)

    def __del__(self):
        self.process.terminate()

    def reset(self):
        msg = (SimulationCommands.reset, None)
        self.pipe.send(msg)
        retr = self.pipe.recv()
        retr = retr.reshape(1, retr.shape[0])
        return retr

    def step(self, action):
        msg = (SimulationCommands.step, action)
        self.pipe.send(msg)
        retr = self.pipe.recv()
        retr = retr.reshape(1, retr.shape[0])
        return retr


class Worker(object):
    def __init__(self, env_name, agent, max_steps=None, batch_size=64, gamma=0.99):
        self.agent = agent
        self.env = IsolatedEnv(env_name)
        # self.ou = OUNoise ( action_dimension=agent.act_space )
        self.t = 0
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.gamma = gamma

    def run(self, sess, coord):
        last_t = 0
        summarize = False
        with sess.as_default(), sess.graph.as_default():
            try:
                while not coord.should_stop():
                    self.agent.sync()
                    state = self.env.reset()

                    state, timesteps, tot_rw = self.sample(state, summarize=summarize)
                    summarize = False
                    self.t += 1

                    if self.agent.name == 'worker_0' and self.t - last_t > 5:
                        last_t = self.t
                        summarize = True
                        tf.logging.info(
                            'Master ep  {}, latest ep reward {}, of steps {}'.format(self.t, tot_rw, timesteps))

                    if self.max_steps is not None and self.t > self.max_steps:
                        tf.logging.info('Hopefully i learnt something...test me...')
                        coord.should_stop()
            except tf.errors.CancelledError:
                return

    def sample(self, state, summarize=False):
        terminal = False
        t, tot_rw = 0, 0

        while not terminal:
            # if stochastic remove exploration noise
            action = self.agent.get_action(state)  # + self.ou.noise()
            next_state, reward, terminal, _ = self.env.step(action)
            self.agent.memory.collect(state, action, reward, next_state, terminal)
            self.agent.think(batch_size=self.batch_size, gamma=self.gamma, summarize=summarize)
            t += 1
            tot_rw += reward

        if summarize:
            ep_summary = tf.Summary()
            ep_summary.value.add(simple_value=tot_rw, tag='eval/total_rw')
            ep_summary.value.add(simple_value=t, tag='eval/ep_length')

            self.agent.writer.add_summary(ep_summary, self.t)
            self.agent.writer.flush()

        return state, t, tot_rw
