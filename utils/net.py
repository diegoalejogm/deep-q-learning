import torch
from torch import nn

import copy
from utils import to_variable, make_dir


class DeepQNetwork(nn.Module):

    def __init__(self, num_actions):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.hidden = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512, bias=True),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(512, num_actions, bias=True)
        )
        # Init with cuda if available
        if torch.cuda.is_available():
            self.cuda()
        self.apply(self.weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        x = self.out(x)
        return x

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # pass
            m.weight.data.normal_(0.0, 0.02)
            # nn.init.xavier_uniform(m.weight)
        if classname.find('Linear') != -1:
            pass
            # m.weight.data.normal_(0.0, 0.02)
            # m.weight.data.fill_(1)
            # nn.init.xavier_uniform(m.weight)
            # m.weight.data.normal_(0.0, 0.008)


def Q_targets(phi_plus1_mb, r_mb, done_mb, model, gamma=0.99):
    '''
    gamma: future reward discount factor
    '''
    # Calculate Q value with given model
    max_Q, argmax_a = model(to_variable(phi_plus1_mb).float()).max(1)
    max_Q = max_Q.detach()
    # Terminates = 0 if ep. teriminates at step j+1, or = 1 otherwise
    target = to_variable(r_mb).float() + (gamma * max_Q) * \
        (1 - to_variable(done_mb).float())
    target = target.unsqueeze(1)
    return target


def Q_values(model, phi_mb, action_mb):
    # Obtain Q values of minibatch
    q_phi = model(to_variable(phi_mb).float())
    # Obtain actions matrix
    action_mb = to_variable(action_mb).long().unsqueeze(1)
    # Select Q values for given actions
    q_phi = q_phi.gather(1, action_mb)
    return q_phi


def gradient_descent(y, q, optimizer):
    # Clear previous gradients before backward pass
    optimizer.zero_grad()

    # Run backward pass
    error = (y - q)

    # Clip error to range [-1, 1]
    error = error.clamp(min=-1, max=1)
    # Square error
    error = error**2
    error = error.sum()
    # q.backward(error.data)
    error.backward()

    # Perfom the update
    optimizer.step()

    return q, error


def save_network(model, episode, out_dir):
    out_dir = '{}/models'.format(out_dir)
    # Make Dir
    make_dir(out_dir)
    # Save model
    torch.save(model.state_dict(), '{}/episode_{}'.format(out_dir, episode))


def copy_network(Q):
    q2 = copy.deepcopy(Q)
    if torch.cuda.is_available():
        return q2.cuda()
    return q2
