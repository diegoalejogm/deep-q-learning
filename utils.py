import os
import errno


def clear_terminal_output():
    os.system('clear')


def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
