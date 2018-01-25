import numpy as np
import json
import utils
import sys

import torch
from torch.autograd import Variable


class ReplayMemory():

    def __init__(self, N, load_existing, data_dir, image_shape=(4, 84, 84)):
        self.data_dir = '{}/replay'.format(data_dir)
        self.memory_size = N
        # Next position in arrays to be used
        self.index = 0

        # One np array for each tuple element
        self.phi_t = np.empty((N, ) + image_shape, dtype=np.float16)
        self.action = np.empty(N, dtype=np.uint8)
        self.reward = np.empty(N, dtype=np.integer)
        self.phi_t_plus1 = np.empty((N, ) + image_shape, dtype=np.float16)
        self.terminates = np.empty(N, dtype=np.bool)

        self.count = 0
        self.last_saved_size = 0

        if load_existing:
            self.load()

    def to_dict(self, saved_size):

        d = {'memory_size': self.memory_size,
             'index': self.index,
             'count': self.count,
             'saved_size': saved_size
             }
        return d

    def print_size(self):
        print('*********')
        print(sys.getsizeof(self.phi_t))
        print(sys.getsizeof(self.action))
        print(sys.getsizeof(self.reward))
        print(sys.getsizeof(self.phi_t_plus1))
        print(sys.getsizeof(self.terminates))
        print('*********')

    def from_dict(self, dictionary):
        self.memory_size = dictionary['memory_size']
        self.index = dictionary['index']
        self.count = dictionary['count']
        self.last_saved_size = dictionary['saved_size']

    def add(self, experience):
        '''
        This operation adds a new experience e, replacing the earliest experience if full.
        '''
        self.phi_t[self.index] = experience[0]
        # print(self.phi_t[self.index])
        self.action[self.index] = experience[1]
        self.reward[self.index] = experience[2]
        self.phi_t_plus1[self.index] = experience[3]
        self.terminates[self.index] = experience[4]

        # Update value of next index
        self.index = (self.index + 1) % self.memory_size
        self.count = min(self.count + 1, self.memory_size)

    def sample(self, size, as_var=True):
        idxs = np.random.choice(self.count, size)
        # Obtain arrays
        phi = self.phi_t[idxs].astype(np.float32)
        actions = self.action[idxs].astype(np.float32)
        rewards = self.reward[idxs].astype(np.float32)
        next_phi = self.phi_t_plus1[idxs].astype(np.float32)
        terminate = (self.terminates[idxs] + 0).astype(np.float32)

        # Return arrays
        return phi, actions, rewards, next_phi, terminate

    def can_sample(self, size):
        return self.count >= size

    def save(self, size):
        # Create out dir
        utils.make_dir(self.data_dir)
        print('Saving Memory data into Replay Memory Instance...')
        # Save property dict
        with open('{}/properties.json'.format(self.data_dir), 'w') as f:
            json.dump(self.to_dict(size), f)
        # Save arrays
        np.save('{}/phi_t'.format(self.data_dir), self.phi_t[: size])
        np.save('{}/action'.format(self.data_dir), self.action[: size])
        np.save('{}/reward'.format(self.data_dir), self.reward[: size])
        np.save('{}/phi_t_plus1'.format(self.data_dir),
                self.phi_t_plus1[:size])
        np.save('{}/terminates'.format(self.data_dir), self.terminates[:size])

    def load(self):
        # Create out dir
        utils.make_dir(self.data_dir)
        try:
            print('Loading Memory data into Replay Memory Instance...')
            # Load property list
            d = json.load(open('{}/properties.json'.format(self.data_dir)))
            self.from_dict(d)
            # Load numpy arrays
            self.phi_t[:self.last_saved_size] = np.load(
                '{}/phi_t.npy'.format(self.data_dir))
            self.action[:self.last_saved_size] = np.load(
                '{}/action.npy'.format(self.data_dir))
            self.reward[:self.last_saved_size] = np.load(
                '{}/reward.npy'.format(self.data_dir))
            self.phi_t_plus1[:self.last_saved_size] = np.load(
                '{}/phi_t_plus1.npy'.format(self.data_dir))
            self.terminates[:self.last_saved_size] = np.load(
                '{}/terminates.npy'.format(self.data_dir))
            print('Finished loading Memory data into Replay Memory Instance!')
        except IOError as e:
            self.__init__(self.memory_size,
                          data_dir=self.data_dir, load_existing=False)
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            print("Couldn't find initial values for Replay Memory, instance init as new.")

    def size(self):
        return (self.phi_t.nbytes + self.action.nbytes + self.reward.nbytes +
                self.phi_t_plus1.nbytes + self.terminates.nbytes)


class History():

    def __init__(self, length=4):
        self.length = length
        self.list = []

    def add(self, ex):
        # Add new element if list is not full
        if len(self.list) < self.length:
            self.list.append(ex)
            return

        # Move existing elements one index to the left
        self.list[:-1] = self.list[1:]
        # Add new value to last index
        self.list[-1] = ex

    def get(self):
        return self.list

    @staticmethod
    def initial(env):
        s = env.reset()[0]
        H = History()
        for _ in range(H.length):
            H.add(s)
        return H
