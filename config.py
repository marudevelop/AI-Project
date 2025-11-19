import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_ROOT = 'data/hmdb51/rawframes'
TRAIN_TXT = 'data/hmdb51/hmdb51_train_split_1_rawframes.txt'
VAL_TXT = 'data/hmdb51/hmdb51_val_split_1_rawframes.txt'
MOTION_JSON = 'data/hmdb51/hmdb51_img_diff.json'

NUM_CLASSES = 51
NUM_SEGMENTS = 8
BATCH_SIZE = 8
NUM_WORKERS = 0
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
