import numpy as np
from utils.running_stats import ZFilter
from osim.env import RunEnv
from collections import deque


class Environment(RunEnv):
    def __init__(self , frame_rate=50 , concat=3 , augment_rw=False , normalize=True):
        super(Environment , self).__init__(visualize=True , max_obstacles=3)
        self.frame_rate = frame_rate
        self.observation_space = (self.observation_space.shape[0] * concat) + 7
        self.sample = self.action_space.sample
        self.action_space , self.bound = (self.action_space.shape[0] , (self.action_space.low , self.action_space.high))
        self.split_obs = self.observation_space - (3 * concat)
        self.difficulty = 3
        self.r = 0
        self.concat = concat
        self.augment_rw = augment_rw
        self.queue = deque(maxlen=self.concat)
        if normalize:
            self.z_filter = ZFilter(self.observation_space)
        else:
            self.z_filter = None

    def get_dims(self):
        return self.observation_space , self.action_space , self.bound

    # def reset(self, difficulty=0):
    def reset(self , difficulty=3 , seed=None):
        state = super(Environment , self).reset(difficulty , seed)

        states = np.tile(state , self.concat)
        self.queue.append(self.get_observation())
        self.r = 0
        return self.concat_frame(states)

    def step(self , action):
        terminal = False
        reward = 0

        for _ in range(self.concat - 1):
            state , r , terminal , info = self.skip_frame(action)
            self.queue.append(state)
            reward += r
            if terminal:
                # pad the queue with end state
                self.queue.append(state)

        self.r += reward
        if self.augment_rw:
            reward += self.surr_rw(self.queue[-1] , action)

        return self.concat_frame(np.array(list(self.queue)).flatten()) , reward , terminal , None

    def skip_frame(self , action):
        reward = 0
        for _ in range(int(100 / self.frame_rate)):
            s , r , t , info = self._step(action)
            reward += r
            if t:
                break

        return s , r , t , info

    def surr_rw(self , state , action):
        state = np.array(state)
        state = self.normalize_cm(state)

        # stay_up
        delta_h = (state[27] - .5 * (state[35] + state[33]))

        # v_pelvis_x - fall_penalty - movement normalized wrt the height - wild actions
        rw = 10 * state[4] - 10 * (delta_h < 0.8) - abs(delta_h - 1.)  # - 0.02 * np.linalg.norm( action )
        return np.asscalar(rw)

    def concat_frame(self , states):
        len = states.shape[0]
        width = len / self.concat
        dest = states.copy()
        last_cols = []
        for idx in range(self.concat):
            state_from = (idx * width)
            state_to = ((idx + 1) * width) - 3

            dest_from = (idx * width) - (3 * idx)
            dest_to = dest_from + (width - 3)

            last_cols.append(states[state_to: state_to + 3])

            dest[dest_from: dest_to] = self.normalize_cm(states[state_from: state_to])

        dest[-3 * self.concat:] = np.concatenate(last_cols)

        last_state_from = (width - 3) * (self.concat - 1)
        vel = self.augment_state(states[:width - 3] , states[last_state_from:])
        states = np.insert(arr=states , obj=(width - 3) , values=vel)

        if self.z_filter is not None:
            states = self.z_filter(states)
        return states

    def update_diff(self , reward):

        if reward > 2:
            self.diff = 1
        elif reward > 3:
            self.diff = 2

    def augment_state(self , s , s1):
        idxs = [22 , 24 , 26 , 28 , 30 , 32 , 34]
        vel = (s1[idxs] - s[idxs]) / (100. / self.frame_rate)
        return vel

    @staticmethod
    def normalize_cm(s):
        # Normalize x,y relative to the torso, and computing relative positon of the center of mass

        torso = [1 , 2 , 4 , 5]
        cm_xy = [18 , 19 , 20 , 21]
        x_pos = [1 , 22 , 24 , 26 , 28 , 30 , 32 , 34]
        y_pos = [2 , 23 , 25 , 27 , 29 , 31 , 33 , 35]

        s[x_pos] = s[x_pos] - s[torso[0]]
        s[y_pos] = s[y_pos] - s[torso[1]]

        s[cm_xy] = s[cm_xy] - s[torso]
        return s
