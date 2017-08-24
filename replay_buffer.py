from collections import deque
import random
import numpy as np
import queue
import heapq


class ReplayBuffer(object):
    def __init__(self, env_shape, buffer_size=1000, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        # self.count = 0
        self.buffer = []
        self.entry = {}
        # heapq.heapify(self.buffer)
        random.seed(random_seed)
        self.env_shape = env_shape
        self.c = 0
        self.alpha = 0.5

    def collect(self, s, a, r, s1, t):

        # collecting trajectory
        s = np.reshape(s, (-1, self.env_shape[0]))
        a = np.reshape(a, (-1, self.env_shape[1]))
        s1 = np.reshape(s1, (-1, self.env_shape[0]))

        entry = (s, a, r, s1, t)
        priority = 1
        res = [priority, self.c]
        self.entry[res[1]] = entry

        # in theory break heap but we rebuild when update
        if self.c == 0:
            self.buffer.append(res)
            self.buffer = np.array(self.buffer)

        self.c += 1
        res = np.reshape(res, newshape=(1, 2))
        #print(self.buffer.shape)
        #print(len(self.entry))
        self.buffer = np.concatenate((self.buffer, res), axis=0)

    def get_size(self):
        return self.buffer.shape[0]

    def update(self, error, ix):
        '''
        update
        '''

        size = int(self.buffer_size) + 1
        tot = sum(self.buffer[:, 0])

        self.buffer[ix, 0] = error
        self.buffer[:, 0] = self.buffer[:, 0] / tot
        self.buffer = self.buffer[self.buffer[:, 0].argsort()]

        old_exp_ix = self.buffer[size:, 1]
        self.buffer = self.buffer[:size]

        for k in old_exp_ix:
            del self.entry[k]

    def get_sample(self, batch_size):

        priority = self.buffer[:, 0]
        priority = priority / sum(priority)

        n = len(priority)

        if n <= batch_size:
            ix = np.random.choice(a=n, size=n, replace=True, p=priority)
        else:
            ix = np.random.choice(a=n, size=batch_size, replace=True, p=priority)

        ref = self.buffer[ix, 1]
        batch = [self.entry[i] for i in ref]
        batch = np.array(batch)

        s1_batch = np.stack(batch[:, 0], axis=0)  # np.vstack(list(map(lambda x: x[0], batch)))
        s1_batch = np.reshape(s1_batch, (-1, self.env_shape[0]))
        a_batch = np.stack(batch[:, 1], axis=0)  # np.vstack(np.array(list(map(lambda x: x[1], batch))))
        a_batch = np.reshape(a_batch, (-1, self.env_shape[1]))
        r_batch = np.stack(batch[:, 2])  # np.vstack(np.array(list(map(lambda x: x[2], batch))))
        s2_batch = np.stack(batch[:, 3], axis=0)  # np.vstack(np.array(list(map(lambda x: x[3], batch))))
        s2_batch = np.reshape(s2_batch, (-1, self.env_shape[0]))
        t_batch = np.stack(batch[:, 4])  # np.vstack(np.array(list(map(lambda x: x[4], batch))))

        return [s1_batch, a_batch, r_batch, s2_batch, t_batch, ix]

    def clear(self):
        self.buffer.clear()
        self.count = 0
