from collections import deque
import random
import numpy as np
import queue
import heapq


class ReplayBuffer(object):
    def __init__(self, env_shape, buffer_size=2000, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        #self.count = 0
        self.buffer = []
        heapq.heapify(self.buffer)
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
        self.c += 1
        # in theory break heap but we rebuild when update
        self.buffer.append((priority, self.c, entry))
        print(len(self.buffer))

    # def add(self, exp):
    #
    #     if self.count < self.buffer_size:
    #         self.buffer.append(exp)
    #         self.count += 1
    #     else:
    #         self.buffer.popleft()
    #         self.buffer.append(exp)

    def get_size(self):
        return len(self.buffer)


    def update(self, error, sample):

        print(len(self.buffer))

        size = int(self.buffer_size)

        values = sorted(self.buffer, reverse=True, key = lambda u:u[0])[:size]

        priority = list(map(lambda x: x[0], values))
        entry = list(map(lambda x: x[2], values))

        priority.extend(error)
        priority = np.array(priority)
        priority = priority / sum(priority)

        entry.extend(sample)

        count = [i for i in range(len(entry))]
        if self.c < len(count):
            self.c = len(count) + 1
        else:
            self.c += 1

        self.buffer = []
        heapq.heapify(self.buffer)
        for i in range(len(entry)):
            self.buffer.append((priority[i], count[i], entry[i]))


    def get_sample(self, batch_size):

        '''
        wrong
        '''

        priority = list(map(lambda x: x[0], self.buffer))
        priority = np.array(priority)
        priority = priority / sum(priority)
        entry = list((map(lambda x: x[2], self.buffer)))

        if len(entry) < batch_size:
            ix = np.random.choice(a=len(entry), size=len(entry), replace=True, p=priority)
        else:
            ix = np.random.choice(a=len(entry), size=batch_size, replace=True, p=priority)

        batch = [i for i in entry if i in ix]
        print(len(batch))
        s1_batch = np.vstack(list(map(lambda x: x[0], batch)))
        a_batch = np.vstack(np.array(list(map(lambda x: x[1], batch))))
        r_batch = np.vstack(np.array(list(map(lambda x: x[2], batch))))
        s2_batch = np.vstack(np.array(list(map(lambda x: x[3], batch))))
        t_batch = np.vstack(np.array(list(map(lambda x: x[4], batch))))

        return s1_batch, a_batch, r_batch, s2_batch, t_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
