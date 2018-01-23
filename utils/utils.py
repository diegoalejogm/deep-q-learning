import os
import errno
import torch
from torch.autograd import Variable


def clear_terminal_output():
    os.system('clear')


def to_variable(arr):
    arr = Variable(torch.from_numpy(arr))
    if torch.cuda.is_available():
        arr = arr.cuda()
    return arr


def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
