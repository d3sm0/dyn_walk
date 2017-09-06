import numpy as np


class Dataset(object):
    def __init__(self , data , shuffle=True):
        self.data = data
        self.enable_shuffle = True
        self.n = next(iter(data.values())).shape[0]
        self._next_id = 0
        self.shuffle()

    def shuffle(self):
        perm = np.arange(self.n)
        for key in self.data:
            self.data[key] = self.data[key][perm]
        self._next_id = 0

    def next_batch(self , batch_size=64):
        if self._next_id >= self.n and self.enable_shuffle:
            self.shuffle()

        curr_id = self._next_id
        curr_batch_size = min(batch_size , self.n - self._next_id)
        self._next_id += curr_batch_size
        data = dict()
        for key in self.data:
            data[key] = self.data[key][curr_id:curr_id + curr_batch_size]
        return data

    def iterate_once(self , batch_size=64):
        if self.enable_shuffle: self.shuffle()
        while self._next_id <= self.n - batch_size:
            yield self.next_batch(batch_size)
        self._next_id = 0


class Memory(object):
    def __init__(self , ob_dim, act_dim , max_steps):
        self.max_steps = max_steps
        self.inits = (np.zeros((ob_dim,)) , np.zeros((act_dim,)))
        self.reset()

    def reset(self):
        ob, act = self.inits
        self.obs = np.array([ob for _ in range(self.max_steps)])
        self.acts = np.array([act for _ in range(self.max_steps)])
        self.rws = np.zeros(self.max_steps , 'float32')
        self.vs = np.zeros(self.max_steps , 'float32')
        self.ds = np.zeros(self.max_steps , 'int32')

    def collect(self , step , t):
        ob , act , r , d , v = step
        t = t % self.max_steps
        self.obs[t] = ob
        self.acts[t] = act
        self.rws[t] = r
        self.vs[t] = v
        self.ds[t] = d

    def release(self , v , done):
        m = {
            'obs': self.obs ,
            'acts': self.acts ,
            'rws': self.rws ,
            'vs': self.vs ,
            'v_next': v * (1 - done) ,
            'ds': self.ds
        }

        return m

    def save(self , m , file_dir):
        try:
            with open(file_dir , 'a') as f:
                f.write(m)
        except IOError:
            print('Memory not saved')
            raise

