import utils
import nets
from logger import Logger
from PIL import Image
import gym

import numpy as np

import torch
from torch.autograd import Variable
from models import History

from nets import DeepQNetwork
from processing import phi_map, tuple_to_numpy


def to_variable(arr):
    v = Variable(torch.from_numpy(arr).float())
    return v


def greedy_action(Q, phi):
    # Otherwise select a_t = argmax_a Q(phi, a)
    phi = to_variable(phi)
    print(phi)
    print(Q(phi))
    return Q(phi).max(1)[1].data


def initial_history(env):
    s = env.reset()[0]
    H = History()
    for _ in range(H.length):
        H.add(s)
    return H
# ----------------------------------


# Play
env = gym.make('Pong-v0')
H = initial_history(env)
Q = DeepQNetwork(6)
Q.load_state_dict(torch.load('data/models/episode_10',
                             map_location=lambda storage, loc: storage))

print(Q.state_dict())
# raw_input()

while(True):
    env.render(mode='human')
    phi = phi_map(H.get())
    action = greedy_action(Q, phi)
    # raw_input()
    image, reward, done, _ = env.step(action)
    H.add(image)
    if done:
        H.add(env.reset()[0])
