from collections import deque
import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, env_shape, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer = deque(buffer_size)
        random.seed(random_seed)
        # np.random.seed(random_seed)
        self.env_shape = env_shape

    def collect(self, s, a, r, s1, t):
        # TODO: check for bottleneck
        # collecting trajectory
        s = np.reshape(s, (-1, self.env_shape[0]))
        a = np.reshape(a, (-1, self.env_shape[1]))
        s1 = np.reshape(s1, (-1, self.env_shape[0]))

        self.add((s, a, r, s1, t))

    def add(self, exp):
        self.buffer.append(exp)

    def get_size(self):
        return len(self.buffer)

    def get_sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)
        batch = np.array(batch)

        # TODO: check for bottleneck
        s1_batch = np.vstack(batch[:, 0])
        a_batch = np.vstack(batch[:, 1])
        r_batch = batch[:, 2]
        s2_batch = np.vstack(batch[:, 3])
        t_batch = batch[:, 4]

        return s1_batch, a_batch, r_batch, s2_batch, t_batch

    def clear(self):
        self.buffer.clear()
