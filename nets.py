import torch
from torch import nn

import copy
import utils


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
            nn.Linear(512, num_actions, bias=True),
            nn.ReLU()
        )
        # Init with cuda if available
        if torch.cuda.is_available():
            self.cuda()
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size()[0], -1)
        x = self.hidden(x)
        x = self.out(x)
        return x


def save_checkpoint(model, episode, out_dir):
    out_dir = '{}/models'.format(out_dir)
    # Make Dir
    utils.make_dir(out_dir)
    # Save model
    torch.save(model.state_dict(), '{}/episode_{}'.format(out_dir, episode))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        # nn.init.xavier_uniform(m.weight)
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        # nn.init.xavier_uniform(m.weight)
        # m.weight.data.normal_(0.0, 0.008)


def update_target(Q):
    q2 = copy.deepcopy(Q)
    if torch.cuda.is_available():
        return q2.cuda()
    return q2
