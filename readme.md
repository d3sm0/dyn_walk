## Deep Deterministic Policy Gradient for Osim rl

Current best model:

ENV_NAME = 'osim'
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 32
GAMMA = 0.99
NUM_EP = 1000
SAVE_EVERY = 100
H_SIZE = [ 128, 128 ]
PRE_TRAIN = None
POLICY = 'det' 
ACTIVATION = lrelu
CONCATENATE_FRAMES  = 2
USE_RW = True
NORMALIZE = False

Test to do:
- H_SIZE = [128,64]
- CONCAT_FRAMES  = 3
- NORMALIZE = True
- PRE_TRAIN = True (500 steps)
- POLICY = 'sin' during pre_training

Stuff to do:

- check train.py