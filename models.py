import numpy as np


class ReplayMemory():

    def __init__(self, N, image_shape=(4, 84, 84)):
        self.N = N
        # Next position in arrays to be used
        self.index = 0

        # One np array for each tuple element
        self.phi_t = np.zeros((N, ) + image_shape, dtype=np.float16)
        self.action = np.zeros(N, dtype=np.uint8)
        self.reward = np.zeros(N, dtype=np.integer)
        self.phi_t_plus1 = np.zeros((N, ) + image_shape, dtype=np.float16)
        self.terminates = np.zeros(N, dtype=np.bool)

        self.count = 0

    def add(self, experience):
        '''
        This operation adds a new experience e, replacing the earliest experience if full.
        '''
        self.phi_t[self.index] = experience[0]
        self.action[self.index] = experience[1]
        self.reward[self.index] = experience[2]
        self.phi_t_plus1[self.index] = experience[3]
        self.terminates[self.index] = experience[4]

        # Update value of next index
        self.index = (self.index + 1) % self.N
        self.count = min(self.count + 1, self.N)

    def sample(self, size):
        idxs = np.random.choice(self.count, size)
        return self.phi_t[idxs].astype(np.float32), self.action[idxs], \
            self.reward[idxs], self.phi_t_plus1[idxs].astype(np.float32), \
            self.terminates[idxs]

    def can_sample(self, size):
        return self.count >= size


class History():

    def __init__(self):
        self.max_size = 4
        self.list = []

    def add(self, ex):
        # Add new element if list is not full
        if len(self.list) < self.max_size:
            self.list.append(ex)
            return

        # Move existing elements one index to the left
        self.list[:-1] = self.list[1:]
        # Add new value to last index
        self.list[-1] = ex

    def get(self):
        return self.list
