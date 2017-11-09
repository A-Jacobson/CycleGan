import os

# DATA
PATH = '/home/austin/data/horse2zebra/'
PATH_A = PATH + 'trainA'
PATH_B = PATH + 'trainB'

# HYPER PARAMS
SCALE = 256
CYCLE_WEIGHT = 10
IDENTITY_WEIGHT = 0
NUM_RESBLOCKS = 9  # set to 6 if 128dim image, 9 if otherwise

# TRAINING PARAMS
EXP_NAME = 'horse2zebra_256_BCE'
NUM_EPOCHS = 20
RESUME_EPOCH = 0
WEIGHTS_DIR = os.path.join('weights', EXP_NAME)
VISUALIZATION_FREQ = 5  # epochs
CHECKPOINT_FREQ = 10  # epochs
