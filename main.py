import utils
import nets
from logger import Logger

import gym

import numpy as np

import torch
from torch import nn, optim
from torch.autograd import Variable

from models import ReplayMemory, History
from nets import DeepQNetwork
from processing import phi_map, tuple_to_numpy


def to_variable(arr):
    v = Variable(torch.from_numpy(arr).float())
    if torch.cuda.is_available():
        return v.cuda()
    return v


def initial_history(env):
    s = env.reset()[0]
    H = History()
    for _ in range(H.length):
        H.add(s)
    return H


def e_greedy_action(Q, phi, env, step):
    # Initial values
    initial_epsilon, final_epsilon = 1.0, .1
    min_eps = 0.1
    # Decay steps
    decay_steps = float(1e7)
    # Calculate annealed epsilon
    step_size = (initial_epsilon - final_epsilon) / decay_steps
    ann_eps = initial_epsilon - step * step_size
    # Calculate epsilon
    epsilon = max(min_eps, ann_eps)
    # Obtain a random value in range [0,1)
    rand = np.random.uniform()
    # With probability e select random action a_t
    if rand < epsilon:
        return env.action_space.sample()

    else:
        # Otherwise select a_t = argmax_a Q(phi, a)
        phi = to_variable(phi)
        return Q(phi).max(1)[1].data


def approximate_targets(phi_plus1_mb, r_mb, done_mb, Q_, gamma=0.99):
    '''
    gamma: future reward discount factor
    '''
    max_Q, argmax_a = Q_(to_variable(phi_plus1_mb)).detach().max(1)
    # 0 if ep. teriminates at step j+1, 1 otherwise
    terminates = to_variable(1 - done_mb)
    return to_variable(r_mb) + (gamma * max_Q) * terminates


def gradient_descent(optimizer, y, Q, phi_mb, action_mb, mb_size):
    # Clear previous gradients before backward pass
    optimizer.zero_grad()

    # Calculate Q(phi) of actions in [action_mb]
    q_phi = Q(to_variable(phi_mb))[np.arange(mb_size), action_mb]

    # Run backward pass
    error = (q_phi - y)
    # Clip error to range [-1, 1]
    error = torch.clamp(error, min=-1, max=1)
    error = error
    error = error.sum()
    error.backward()

    # Perfom the update
    optimizer.step()

    return q_phi, error


# Tranining
env = gym.make('Pong-v0')
# Current iteration
step = 0
# Has trained model
has_trained_model = False
# Init training params
params = {
    'num_episodes': 100,
    'minibatch_size': 32,
    'max_episode_length': int(10e6),  # T
    'memory_size': int(1e6),  # N
    'history_size': 4,  # k
    'train_freq': 4,
    'target_update_freq': 10000,  # C: Target nerwork update frequency
    'num_actions': env.action_space.n,
    'min_steps_train': 50000
}
# Initialize Logger
log = Logger()
# Initialize replay memory D to capacity N
D = ReplayMemory(params['memory_size'], load_existing=True)
skip_fill_memory = D.count > 0
# Initialize action-value function Q with random weights
Q = DeepQNetwork(params['num_actions'])
# Initialize target action-value function Q^ with weights
Q_ = nets.update_target(Q)
# Init network optimizer
optimizer = optim.RMSprop(
    Q.parameters(), lr=0.00025, momentum=0.95, alpha=0.95, eps=.01
)

# Initialize sequence s1 = {x1} and preprocessed sequence phi = phi(s1)
H = initial_history(env)

for ep in range(params['num_episodes']):

    phi = phi_map(H.get())

    for _ in range(params['max_episode_length']):
        env.render(mode='human')
        step += 1
        # Select action
        action = e_greedy_action(Q, phi, env, step)
        # Execute action a_t in emulator and observe reward r_t and image x_(t+1)
        image, reward, done, _ = env.step(action)
        # Set s_(t+1) = s_t, a_t, x_(t+1) and preprocess phi_(t+1) =  phi_map( s_(t+1) )
        H.add(image)
        phi_prev, phi = phi, phi_map(H.get())
        # Store transition (phi_t, a_t, r_t, phi_(t+1)) in D
        D.add((phi_prev, action, reward, phi, done))

        should_train_model = step > 30  # (step > params['min_steps_train']) and \
        #D.can_sample(MINIBATCH_SIZE) and (t % params['train_freq'] == 0)

        if should_train_model:
            if not (skip_fill_memory or has_trained_model):
                D.save(params['min_steps_train'])
                has_trained_model = True

            # Sample random minibatch of transitions ( phi_j, a_j, r_j, phi_(j+1)) from D
            phi_mb, a_mb, r_mb, phi_plus1_mb, done_mb = D.sample(
                params['minibatch_size'])
            # Set y_j
            y = approximate_targets(phi_plus1_mb, r_mb, done_mb, Q_)
            # Perform a gradient descent step on ( y_j - Q(phi_j, a_j) )^2
            q_phi, loss = gradient_descent(optimizer, y, Q,
                                           phi_mb, a_mb, params['minibatch_size'])
            # Reset Q_
            if step % params['target_update_freq'] == 0:
                Q_ = nets.update_target(Q)

            log.update(reward, q_phi, loss)

        # Restart game if done
        if done:
            H.add(env.reset()[0])

    if ep % 25 == 0:
        nets.save_checkpoint(Q, ep)
writer.close()
