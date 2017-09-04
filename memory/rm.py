from collections import deque
import random
import numpy as np
import pickle
import os
from datetime import datetime
class Experience ( object ):
    def __init__(self , buffer_size ,batch_size, log_dir):
        """
        The right side of the deque contains the most recent experiences
        """
        self.count = 0
        self.buffer = deque (maxlen=buffer_size)
        self.batch_size = batch_size
        now = datetime.utcnow().strftime("%b-%d_%H_%M")  # create unique dir
        self.log_dir = os.path.join(log_dir, 'dataset')
        try:
            os.mkdir(self.log_dir)
        except OSError:
            pass
    def collect(self , step, curr_ep):

        if len(self.buffer) < self.buffer.maxlen:
            self.buffer.append ( step )
            self.count += 1
        else:
            self.save(curr_ep)
            self.clear()
            self.buffer.append(step)

    def save(self, curr_ep):
        file_name = 'dump_{}'.format(curr_ep)
        if os.path.isfile(os.path.join(self.log_dir , file_name)):
            now = datetime.utcnow().strftime("%b-%d_%H_%M")  # create unique dir
            new_dir = os.path.join(self.log_dir, now)
            os.mkdir(new_dir)
            self.log_dir = new_dir
        try:
            with open(os.path.join(self.log_dir, file_name), 'ab') as f:
                pickle.dump(self.buffer, f)
        except IOError:
            raise
    def get_size(self):
        return len(self.buffer)

    def select(self ):

        if self.count < self.batch_size:
            batch = random.sample ( self.buffer , self.count )
        else:
            batch = random.sample ( self.buffer , self.batch_size )

        batch = np.array ( batch )
        s1_batch = np.vstack ( batch[ : , 0 ] )
        a_batch = np.vstack ( batch[ : , 1 ] )
        r_batch = batch[ : , 2 ]
        s2_batch = np.vstack ( batch[ : , 3 ] )
        t_batch = batch[ : , 4 ]

        return s1_batch , a_batch , r_batch , s2_batch , t_batch

    def clear(self):
        self.buffer.clear ()
        self.count = 0