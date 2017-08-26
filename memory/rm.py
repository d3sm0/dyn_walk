from collections import deque
import random
import numpy as np

class Experience ( object ):
    def __init__(self , buffer_size ,batch_size):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque ()
        self.batch_size = batch_size

        # random.seed ( random_seed )


    def add(self , s , a , r , s1 , t):

        # collecting trajectory
        s = np.reshape ( s , (1 , -1))
        a = np.reshape ( a , (1,-1))
        s1 = np.reshape ( s1 , (1,-1))
        
        self.collect ( (s , a , r , s1 , t) )


    def collect(self , exp):

        if self.count < self.buffer_size:
            self.buffer.append ( exp )
            self.count += 1
        else:
            self.buffer.popleft ()
            self.buffer.append ( exp )

    def get_size(self):
        return self.count

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