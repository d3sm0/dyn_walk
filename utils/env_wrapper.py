import numpy as np

# TODO rewrite observation management using FIFO or queue
class EnvWrapper( object ):
    def __init__(self , Env , visualize=False , frame_rate=50 , concat=3 , augment_rw=False , normalize=True ,
                 add_acceleration=5 , add_time=True):
        self.env = Env( visualize=visualize )
        self.frame_rate = frame_rate
        self.observation_space = self.env.observation_space.shape[ 0 ] * concat + (add_acceleration + bool( add_time ))
        self.action_space = self.env.action_space.shape[ 0 ]
        self.bound = (self.env.action_space.low , self.env.action_space.high)
        self.split_obs = self.observation_space - (3 * concat) - bool( add_time )
        self.difficulty = 0
        self.r = 0
        self.concat = concat
        self.augment_rw = augment_rw
        self.add_acceleration = add_acceleration
        self.normalize = normalize

    def get_dims(self):
        return (self.observation_space , self.action_space , self.bound)

    def close(self):
        return self.env.close()

    def reset(self , difficulty=0):
        state = self.env.reset( difficulty )
        states = np.tile( state , self.concat )
        self.r = 0
        return self.concat_frame( states )

    def step(self , action):


        states = [ self.env.get_observation() ]

        reward = 0
        for _ in range( self.concat - 1 ):
            state , r , terminal , info = self.skip_frame( action )
            states.append( state )
            reward += r

            if terminal and len(states) < self.concat:
                states.append( states[ -1 ] )
                break

        self.r += reward

        if self.augment_rw:
            reward += self.surr_rw( state , action )

        return self.concat_frame( states ) , reward , terminal , info

    def skip_frame(self , action):
        reward = 0
        for _ in range( int( 100 / self.frame_rate ) ):
            s , r , t , info = self.env.step( action )
            reward += r
            if t:
                break

        return s , r , t , info

    def surr_rw(self , state , action):
        state = np.array( state )
        delta_h = state[ 27 ] - state[ 35 ]
        rw = 10 * state[ 20 ] - abs( delta_h - 1.2 ) - 0.1 * np.linalg.norm( action ) - 10 * (state[ 27 ] < 0.8)
        return np.asscalar( rw )

    def concat_frame(self , states):

        states = np.reshape( states , (self.concat , -1) )
        state_list = [ ]

        state_list.append( np.append( states[ : , :38 ] , states[ : , 38: ] ) )

        if self.add_acceleration is not None:
            state_list[ 0 ] = self.augment_state( state_list[ 0 ] , state_list[ -1 ] )
        if self.normalize:
            state_list = [ self.normalize_cm( state ) for state in state_list ]

        return np.array( state_list ).flatten()

    def update_diff(self , reward):

        if reward > 2:
            self.diff = 1
        elif reward > 3:
            self.diff = 2

    def augment_state(self , s , s1):

        s = np.array( s )
        s1 = np.array( s1 )

        idxs = [ 22 , 24 , 26 , 28 , 30 ]

        vel = (s1[ idxs ] - s[ idxs ]) / (100 / self.frame_rate)

        s = np.insert( arr=s , obj=38 , values=vel )
        s = np.insert( arr=s , obj=len( s ) , values=self.env.istep )

        return s

    def normalize_cm(self , s):

        # Normalize x,y relative to the torso, and computing relative positon of the center of mass

        torso = [ 1 , 2 , 4 , 5 ]
        cm_xy = [ 18 , 19 , 20 , 21 ]
        x_pos = [ 1 , 22 , 24 , 26 , 28 , 30 , 32 , 34 ]
        y_pos = [ 2 , 23 , 25 , 27 , 29 , 31 , 33 , 35 ]

        s[ x_pos ] = s[ x_pos ] - s[ 1 ]
        s[ y_pos ] = s[ y_pos ] - s[ 2 ]

        s[ cm_xy ] = s[ cm_xy ] - s[ torso ]

        return s
