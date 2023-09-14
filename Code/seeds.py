import os
import random
import numpy as np
import torch

DEFAULT_SEED = 1337

def set_python(seed=DEFAULT_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

def set_numpy(seed=DEFAULT_SEED):
    np.random.seed(seed)

def set_torch(seed=DEFAULT_SEED, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def set_all_seeds(seed=DEFAULT_SEED, deterministic=False):
    set_python(seed)
    set_numpy(seed)
    set_torch(seed, deterministic)

if __name__ == '__main__':
    set_all_seeds()