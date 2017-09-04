import numpy as np
from utils.running_stats import ZFilter

# TODO rewrite observation management using FIFO or queue


class EnvWrapper( object ):
    def __init__(self , Env , visualize=False , frame_rate=50 , concat=3 , augment_rw=False , normalize=True ,
                 add_acceleration=7 , add_time=False, z_filter = None):

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
        self.z_filter = ZFilter(self.observation_space)

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

            if terminal and len( states ) < self.concat:
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

        # state = self.normalize_cm( state )

        # stay_up
        delta_h = (state[ 27 ] - .5 * (state[ 35 ] + state[ 33 ]))
        delta_x = (state[ 18 ] - .5 * (state[ 32 ] + state[ 34 ]))

        # v_pelvis_x - fall_penalty - movement normalized wrt the height - wild actions
        # rw = 10 * state[ 4 ] - 10 * (state[2] < 0.65) # - abs( delta_h - 1. )  # - 0.02 * np.linalg.norm( action )
        # rw = 10 * (1 - 2 * max( 0 , (abs( delta_x - 0.1 ) - 0.15) ))  - 10*(delta_h < 0.7)
        rw = 10 * state[ 4 ] - 10 * (delta_h < 0.8) - abs( delta_h - 1. )  # - 0.02 * np.linalg.norm( action )
        return np.asscalar( rw )

    def concat_frame(self , states):

        # TODO redo with indexes...

        states = np.reshape( states , (self.concat , -1) )

        states = np.append( np.concatenate( states[ : , :38 ] ) ,
                            np.concatenate( states[ : , 38: ] ) )

        if self.normalize:
            states = np.reshape( states , (self.concat , -1) )
            states = np.apply_along_axis( self.normalize_cm , 1 , states )

        if self.add_acceleration is not None:
            states = states.flatten()
            vel = self.augment_state( states[ :38 ] , states[ 41 * (self.concat - 1):41 * self.concat - 3 ] )
            states = np.insert( arr=states , obj=38 , values=vel )

        if self.z_filter is not None:
            states = self.z_filter(states)

        return states

    def update_diff(self , reward):

        if reward > 2:
            self.diff = 1
        elif reward > 3:
            self.diff = 2

    def augment_state(self , s , s1):

        s = np.array( s )
        s1 = np.array( s1 )

        idxs = [ 22 , 24 , 26 , 28 , 30 , 32 , 34 ]

        vel = (s1[ idxs ] - s[ idxs ]) / (100. / self.frame_rate)
        # vel =  np.append(vel, self.env.istep)

        return vel

    def normalize_cm(self , s):

        s = np.array( s )
        # Normalize x,y relative to the torso, and computing relative positon of the center of mass

        torso = [ 1 , 2 , 4 , 5 ]
        cm_xy = [ 18 , 19 , 20 , 21 ]
        x_pos = [ 1 , 22 , 24 , 26 , 28 , 30 , 32 , 34 ]
        y_pos = [ 2 , 23 , 25 , 27 , 29 , 31 , 33 , 35 ]

        s[ x_pos ] = s[ x_pos ] - s[ 1 ]
        s[ y_pos ] = s[ y_pos ] - s[ 2 ]

        s[ cm_xy ] = s[ cm_xy ] - s[ torso ]

        return s
