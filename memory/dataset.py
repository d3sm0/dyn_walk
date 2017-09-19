import numpy as np
import pickle
import os


class Dataset(object):
    def __init__(self, data, batch_size=64, shuffle=True):
        self.data = data
        self.enable_shuffle = True
        self.n = next(iter(data.values())).shape[0]
        self._next_id = 0
        self.batch_size = batch_size
        if self.enable_shuffle: self.shuffle()

    def shuffle(self):
        perm = np.arange(self.n)
        np.random.shuffle(perm)
        for key in self.data.keys():
            self.data[key] = self.data[key][perm]
        self._next_id = 0

    def next_batch(self):
        if self._next_id >= self.n and self.enable_shuffle:
            self.shuffle()

        curr_id = self._next_id
        curr_batch_size = min(self.batch_size, self.n - self._next_id)
        self._next_id += curr_batch_size
        data = dict()
        for key in self.data.keys():
            data[key] = self.data[key][curr_id:curr_id + curr_batch_size]
        return data

    def iterate_once(self):
        if self.enable_shuffle:
            self.shuffle()
        while self._next_id <= self.n - self.batch_size:
            yield self.next_batch()
        self._next_id = 0


class Memory(object):
    def __init__(self, obs_dim, act_dim, max_steps, main_path):
        self.max_steps = max_steps
        self.inits = (np.zeros((obs_dim,)), np.zeros((act_dim,)))
        try:
            os.makedirs(os.path.join(main_path, 'dataset'))
        except:
            pass
        self.log_dir = os.path.join(main_path, 'dataset')
        self.reset()

    def reset(self):
        ob, act = self.inits
        self.obs = np.array([ob for _ in range(self.max_steps)])
        self.acts = np.array([act for _ in range(self.max_steps)])
        self.rws = np.zeros(self.max_steps, 'float32')
        self.vs = np.zeros(self.max_steps, 'float32')
        self.ds = np.zeros(self.max_steps, 'int32')

    def collect(self, step, t):
        ob, act, r, d, v = step
        i = t % self.max_steps
        self.obs[i] = ob
        self.acts[i] = act
        self.rws[i] = r
        self.vs[i] = v
        self.ds[i] = d

    def release(self, v, done, t):
        m = {
            'obs': self.obs,
            'acts': self.acts,
            'rws': self.rws,
            'vs': self.vs,
            'v_next': v * (1 - done),
            'ds': self.ds
        }
        self.save(m, t)
        # TODO is this needed?
        # self.reset()
        return m

    def save(self, memory, time_steps):
        file_name = 'dump_{}'.format(time_steps)
        try:
            with open(os.path.join(self.log_dir, file_name), 'ab') as f:
                pickle.dump(memory, f)
        except IOError:
            raise
