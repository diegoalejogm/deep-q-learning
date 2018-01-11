import gym

import os
import copy
import errno
import pickle

import numpy as np

import torch
from torch import nn, optim
from torch.autograd import Variable

from models import ReplayMemory, History
from nets import DeepQNetwork
from processing import phi_map, tuple_to_numpy


# Additional Functionality

def clear_output():
    os.system('clear')


def _make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_checkpoint(model, episode, data_folder):
    out_dir = '{}/models'.format(data_folder)
    _make_dir(out_dir)
    torch.save(model.state_dict(), '{}/episode_{}'.format(out_dir, episode))


def to_variable(arr):
    v = Variable(torch.from_numpy(arr).float())
    if torch.cuda.is_available():
        return v.cuda()
    return v


def initial_history(env):
    s = env.reset()[0]
    H = History()
    for _ in range(H.max_size):
        H.add(s)
    return H


def e_greedy_action(Q, phi, env, frame_count, epsilon=None):
    # Calculate annealed epsilon
    initial_epsilon, final_epsilon = 1.0, .1
    max_frames = float(1e7)
    if epsilon is None:
        epsilon = max(final_epsilon, initial_epsilon - frame_count *
                      ((initial_epsilon - final_epsilon) / max_frames))
    print('Epsilon: {}'.format(epsilon))
    # Obtain a random value in range [0,1)
    rand = np.random.uniform()
    # With probability e select random action a_t
    if rand < epsilon:
        return env.action_space.sample()
    # Otherwise select a_t = argmax_a Q(phi, a)
    else:
        # Convert to Variable
        phi = to_variable(phi)
        return Q(phi).max(1)[1].data


def update_target_network(Q,):
    q2 = copy.deepcopy(Q)
    if torch.cuda.is_available():
        return q2.cuda()
    return q2


def approximate_targets(phi_plus1_mb, r_mb, done_mb, Q_, gamma=0.99):
    '''
    gamma: future reward discount factor
    '''
    max_Q, argmax_a = Q_(to_variable(phi_plus1_mb)).detach().max(1)
    # 0 if ep. teriminates at step j+1, 1 otherwise
    terminates = to_variable(1 - done_mb)
    return to_variable(r_mb) + (gamma * max_Q) * terminates


def gradient_descent(optimizer, loss_func, y, Q, phi_mb, action_mb, mb_size):
    # Calculate Q(phi) of actions in [action_mb]
    q_phi = Q(to_variable(phi_mb))[np.arange(mb_size), action_mb]
    # Clip error to range [-1, 1]
#    error = ( torch.clamp(y - q_phi, min=-1, max=1) )**2

    # Clear previous gradients before backward pass
    optimizer.zero_grad()

    # Run backward pass
    error = loss_func(q_phi, y)
    error = torch.clamp(error, min=-1, max=1)**2
    error = error.sum()
    error.backward()

    # Perfom the update
    optimizer.step()

    del q_phi, error


env = gym.make('Pong-v0')
NUM_EPISODES = 1000
MINIBATCH_SIZE = 32
T = 10000000
N = int(1e6)  # int(1e6)  # Replay Memory size: 1M
C = 10000  # Target nerwork update frequency
k = 4  # Agent History Length
t = 0
frame_count = 0
initial_stored = False
ep_reward_list = []
loss_func = nn.MSELoss(size_average=False, reduce=False)


# Initialize replay memory D to capacity N
D = ReplayMemory(N)
# Initialize action-value function Q with random weights
Q = DeepQNetwork(6)
if torch.cuda.is_available():
    Q.cuda()
optimizer = optim.RMSprop(
    Q.parameters(), lr=0.00025, momentum=0.95, alpha=0.95, eps=.01
)
# Initialize target action-value function Q^ with weights
Q_ = update_target_network(Q)

for ep in range(NUM_EPISODES):
    # Initialize sequence s1 = {x1} and preprocessed sequence phi = phi(s1)
    H = initial_history(env)
    phi = phi_map(H.get())

    ep_reward = 0.0
    ep_num_rewards = 0.0

    for _ in range(T):
        t += 1
#         env.render(mode='human')
        # Select action
        action = e_greedy_action(Q, phi, env, frame_count)
        # Execute action a_t in emulator and observe reward r_t and image x_(t+1)
        image, reward, done, _ = env.step(action)
        if reward != 0:
            ep_num_rewards += 1
            # (ep_reward * (ep_num_rewards-1) + reward) / (ep_num_rewards)
            ep_reward += reward
        frame_count += 1
        if done:
            break
        # Set s_(t+1) = s_t, a_t, x_(t+1) and preprocess phi_(t+1) =  phi_map( s_(t+1) )
        H.add(image)
        phi_prev, phi = phi, phi_map(H.get())
        # Store transition (phi_t, a_t, r_t, phi_(t+1)) in D
        D.add((phi_prev, action, reward, phi, done))
        if t > 50000 and D.can_sample(MINIBATCH_SIZE) and t % 4 == 0:
            # Sample random minibatch of transitions ( phi_j, a_j, r_j, phi_(j+1)) from D
            phi_mb, a_mb, r_mb, phi_plus1_mb, done_mb = D.sample(
                MINIBATCH_SIZE)
            # Set y_j
            y = approximate_targets(phi_plus1_mb, r_mb, done_mb, Q_)
            # Perform a gradient descent step on ( y_j - Q(phi_j, a_j) )^2
            gradient_descent(optimizer, loss_func, y, Q,
                             phi_mb, a_mb, MINIBATCH_SIZE)
            # Reset Q_
            if t % C == 0:
                Q_ = update_target_network(Q)

            del y
        # -- LOGS
        clear_output()
        print('Frame #: {}'.format(frame_count))
        print('Ep. {} Reward: {}'.format(ep, ep_reward))
        print('Eps. Rewards: {}'.format(ep_reward_list[:20]))
        print('Replay Size: {}'.format(D.count))
#        print('History Size: {}'.format(len(H.get())))

        # -- \LOGS
        # Restart game if done
        if done:
            break
    if ep % 1 == 0:
        save_checkpoint(Q, ep, './data')
    ep_reward_list.append(ep_reward)
