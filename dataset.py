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
