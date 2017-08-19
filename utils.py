"""
TODO:
- be able to switch between gym and osim.env
- when env.reset() pass a new difficulty value
- integrate new reward function if required by spec (if augmnet_reward= True =>)
- make an assert if the library is present in the environemnt
- whatever you think is handy
"""

from osim.env import RunEnv

env = RunEnv(visualize=True)
observation = env.reset(difficulty = 0)
for i in range(200):
    observation , reward , done , info = env.step ( env.actiozn_space.sample () )
    pass



def surr_rw(state, act):
    delta_h = state[:,27] - state[:,35]
    rw = 10 * state[:,20] - abs(delta_h - 1.2) - 0.1 * np.linalg.norm(act) - 10 * (state[:,27] < 0.8)
    return np.asscalar(rw)
