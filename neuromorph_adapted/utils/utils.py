import os
import math
import numpy as np
from itertools import cycle

def getFiles(targetdir):
    ls = []
    for fname in os.listdir(targetdir):
        path = os.path.join(targetdir, fname)
        if os.path.isdir(path):
            continue
        ls.append(fname)
    return sorted(ls)

def getDirs(parent_dir):
    ls = []
    for dir_name in os.listdir(parent_dir):
        path = os.path.join(parent_dir, dir_name)
        if os.path.isdir(path):
            ls.append(dir_name)
    return sorted(ls)  

class RunningAverage:
    # Computes and stores the average
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.old_avg = 0

    def update(self, value, n=1):
        """
        Use Welford's method to compute running average
        Description here: https://www.johndcook.com/blog/standard_deviation/
        Welford's method is more numerically stable, without requiring two passes
        """
        self.count += n
        if self.count == 1:
            self.avg = self.old_avg = value
        else:
            self.avg = self.old_avg + (n*value - n*self.old_avg)/self.count
            self.old_avg = self.avg

def k_fold_split_train_val_test(dataset_size, fold_num, seed=230597):
    k = int(fold_num-1)
    train_ims, val_ims, test_ims = math.floor(dataset_size*0.7), math.floor(dataset_size*0.1), math.ceil(dataset_size*0.2)
    if dataset_size - (train_ims+val_ims+test_ims) == 1:
        val_ims += 1 # put the extra into val set
    try:
        assert(train_ims+val_ims+test_ims == dataset_size)
    except AssertionError:
        print("Check the k fold data splitting, something's dodgy...")
        exit(1)
    train_inds, val_inds, test_inds = [], [], []
    # initial shuffle
    np.random.seed(seed)
    shuffled_ind_list = np.random.permutation(dataset_size)
    # allocate dataset indices based upon the fold number --> not the prettiest or most efficient implementation, but functional
    cyclic_ind_list = cycle(shuffled_ind_list)
    for i in range(k*test_ims):
        next(cyclic_ind_list)   # shift start pos
    for i in range(test_ims):
        test_inds.append(next(cyclic_ind_list))
    for i in range(train_ims):
        train_inds.append(next(cyclic_ind_list))
    for i in range(val_ims):
        val_inds.append(next(cyclic_ind_list))
    return train_inds, val_inds, test_inds


class ParametersBase:
    """Base class for parameters"""

    def from_dict(self, d):
        for key in d:
            if hasattr(self, key):
                self.__setattr__(key, d[key])

    def print_self(self):
        print("parameters: ")
        p_d = self.__dict__
        for k in p_d:
            print(k, ": ", p_d[k], "  ", end="")
        print("")