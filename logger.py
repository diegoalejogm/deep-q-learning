from tensorboardX import SummaryWriter


class Logger:

    def __init__(self, log_dir):
        self.env_name = 'Pong-v0'
        # TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir)
        # Episode Values
        self.ep = 0
        self.ep_rewards = []
        self.ep_max_reward = 0.0
        self.ep_min_reward = 0.0
        # Updates Values
        self.grad_count = int(0)
        self.total_q = 0.0
        self.total_loss = 0.0
        self.mb_loss = 0.0
        self.mb_q = 0.0

        # Counters
        self.epsilon_val = 0.0
        self.update_count = 0.0
        self.step = 0.0

    def epsilon(self, eps):
        self.step += 1
        self.epsilon_val = eps
        self._log('epsilon', self.epsilon_val, self.step)

    def network(self, net):
        for name, param in net.named_parameters():
            self._log(name, param.clone().cpu().data.numpy(),
                      self.step, type='histogram')

    def q_loss(self, q, loss):
        self.update_count += 1

        self.mb_loss = loss.data.cpu().numpy()
        self.mb_q = q.sum().data.cpu().numpy() / int(q.size()[0])

        self.total_q += self.mb_q[0]
        self.total_loss += self.mb_loss[0]

        avg_loss = self.total_loss / self.update_count
        avg_q = self.total_q / self.update_count

        self._log('update.average_q', avg_q, self.update_count)
        self._log('update.average_loss', avg_loss, self.update_count)
        self._log('update.minibatch_loss', self.mb_loss, self.update_count)
        self._log('update.minibatch_q', self.mb_q, self.update_count)

    def episode(self, reward):
        self.ep_rewards.append(reward)
        self.ep_max_reward = max(self.ep_max_reward, reward)
        self.ep_min_reward = min(self.ep_min_reward, reward)

    def display(self):
        values = {
            'Episode': self.ep,
            'Step': self.step,
            'Avg. Loss': self.total_loss / self.update_count,
            'Avg. Q': self.total_q / self.update_count,
            'Episode Avg. Reward': sum(self.ep_rewards) / float(len(self.ep_rewards)),
            'Episode Min. Reward': self.ep_min_reward,
            'Episode Max. Reward': self.ep_max_reward,
            'Minibatch Loss': self.mb_loss[0],
            'Minibatch Q': self.mb_q[0],
            'Epsilon': self.epsilon_val
        }
        print('-------')
        for key in values:
            print('{}: {}'.format(key, values[key]))

    def reset_episode(self):
        avg_ep_reward = sum(self.ep_rewards) / float(len(self.ep_rewards))

        self._log('ep.average_reward', avg_ep_reward, self.ep)
        self._log('ep.min_reward', self.ep_min_reward, self.ep)
        self._log('ep.max_reward', self.ep_max_reward, self.ep)

        self.ep += 1
        self.ep_rewards = []
        self.ep_max_reward = 0.0
        self.ep_min_reward = 0.0

    def _log(self, name, value, step, type='scalar'):
        # Add Env.Name to name
        name = '{}/{}'.format(self.env_name, name)
        # Log in Tensorboard
        if type == 'scalar':
            self.writer.add_scalar(name, value, step)
        elif type == 'histogram':
            self.writer.add_histogram(name, value, step)
