import numpy as np
import json
import utils
import sys

import torch
from torch.autograd import Variable


class ReplayMemory():

    def __init__(self, N, load_existing, data_dir, image_shape=(4, 84, 84)):
        # Path where replay memory can be stored/loaded from
        self.data_dir = '{}/replay'.format(data_dir)
        # Max memory size
        self.memory_size = N
        # Shape of images
        self.image_shape = image_shape
        # Next position in arrays to be used
        self.index = 0
        # Number of stored elements
        self.count = 0

        # One np array for each tuple element
        self._init_arrays(self.memory_size)

        # Loads existing values if possible
        if load_existing:
            self.load()

    def _init_arrays(self, N):
        '''
        Inits memory arrays as empty numpy arrays with expected shapes
        '''
        self.phi_t = np.empty((N, ) + self.image_shape, dtype=np.float16)
        self.action = np.empty(N, dtype=np.uint8)
        self.reward = np.empty(N, dtype=np.integer)
        self.phi_t_plus1 = np.empty((N, ) + self.image_shape, dtype=np.float16)
        self.terminates = np.empty(N, dtype=np.bool)

    def _save_arrays(self, path, size):
        '''
        Saves slice of current memory arrays with given size into input path
        '''
        np.save('{}/phi_t'.format(path), self.phi_t[: size])
        np.save('{}/action'.format(path), self.action[: size])
        np.save('{}/reward'.format(path), self.reward[: size])
        np.save('{}/phi_t_plus1'.format(path), self.phi_t_plus1[:size])
        np.save('{}/terminates'.format(path), self.terminates[:size])

    def _load_arrays(self, path, size):
        '''
        Loads memory arrays from path with input size
        '''
        if size > self.memory_size:
            message = 'Stored memory size = {} is bigger than actual memory size = {}'.format(
                size, self.memory_size)
            raise ValueError(message)

        self.phi_t[:size] = np.load('{}/phi_t.npy'.format(path))
        self.action[:size] = np.load('{}/action.npy'.format(path))
        self.reward[:size] = np.load('{}/reward.npy'.format(path))
        self.phi_t_plus1[:size] = np.load('{}/phi_t_plus1.npy'.format(path))
        self.terminates[:size] = np.load('{}/terminates.npy'.format(path))
        self.index = size
        self.count = size

    def to_dict(self, saved_size):
        '''
        Converts current replay memory into dict representation
        '''
        d = {
            'saved_size': saved_size
        }
        return d

    def add(self, experience):
        '''
        This operation adds a new experience e, replacing the earliest experience if arrays are full.
        '''
        self.phi_t[self.index] = experience[0]
        self.action[self.index] = experience[1]
        self.reward[self.index] = experience[2]
        self.phi_t_plus1[self.index] = experience[3]
        self.terminates[self.index] = experience[4]

        # Update value of next index
        self.index = (self.index + 1) % self.memory_size
        # Update the count value
        self.count = min(self.count + 1, self.memory_size)

    def sample(self, size):
        '''
        Samples slice of arrays with input size.
        '''
        idxs = np.random.choice(self.count, size)
        # Obtain arrays
        phi = self.phi_t[idxs].astype(np.float32)
        actions = self.action[idxs].astype(np.float32)
        rewards = self.reward[idxs].astype(np.float32)
        next_phi = self.phi_t_plus1[idxs].astype(np.float32)
        terminate = (self.terminates[idxs] + 0).astype(np.float32)

        # Return arrays
        return phi, actions, rewards, next_phi, terminate

    def can_sample(self, sample_size):
        '''
        Returns true if item count is at least as big as sample size.
        '''
        return self.count >= sample_size

    def save(self, size):
        '''
        Saves replay memory (attributes and arrays).
        '''
        # Create out dir
        utils.make_dir(self.data_dir)
        print('Saving Memory data into Replay Memory Instance...')
        # Save property dict
        with open('{}/properties.json'.format(self.data_dir), 'w') as f:
            json.dump(self.to_dict(size), f)
        # Save arrays
        self._save_arrays(self.data_dir, size)

    def load(self):
        '''
        Loads replay memory (attributes and arrays) into self, if possible.
        '''
        # Create out dir
        utils.make_dir(self.data_dir)
        try:
            print('Loading Memory data into Replay Memory Instance...')
            # Load property list
            d = json.load(open('{}/properties.json'.format(self.data_dir)))
            # Load numpy arrays
            self._load_arrays(self.data_dir, d['saved_size'])

            print('Finished loading Memory data into Replay Memory Instance!')

        except IOError as e:
            self.__init__(self.memory_size,
                          data_dir=self.data_dir, load_existing=False)
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            print("Couldn't find initial values for Replay Memory, instance init as new.")

    def arrays_size(self):
        '''
        Returns the size in bytes of the memory's array
        '''
        return (self.phi_t.nbytes + self.action.nbytes + self.reward.nbytes +
                self.phi_t_plus1.nbytes + self.terminates.nbytes)


class History():

    def __init__(self, length=4):
        self.length = length
        self.list = []

    def add(self, ex):
        '''
        Add new element to list if it is not full. Replaces last otherwise.
        '''
        if len(self.list) < self.length:
            self.list.append(ex)
            return

        # Move existing elements one index to the left
        self.list[:-1] = self.list[1:]
        # Add new value to last index
        self.list[-1] = ex

    def get(self):
        '''
        Returns a copy of the list
        '''
        return self.list

    @staticmethod
    def initial(env):
        '''
        Creates a new History with the first state of the input env.
        Repeats initial state to fill list.
        '''
        s = env.reset()[0]
        H = History()
        for _ in range(H.length):
            H.add(s)
        return H
