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

    def network(self, net):
        for name, param in net.named_parameters():
            self._log(name, param.clone().cpu().data.numpy(),
                      self.step, type='histogram')

    def epsilon(self, eps, step):
        self.step = step
        self.epsilon_val = eps
        self._log('epsilon', self.epsilon_val, self.step)

    def q_loss(self, q, loss, step):
        self.step = step
        self.update_count += 1

        self.mb_loss = loss.data.cpu().sum()
        self.mb_q = q.sum().data.cpu().sum() / int(q.size()[0])

        self.total_q += self.mb_q
        self.total_loss += self.mb_loss

        avg_loss = self.total_loss / self.update_count
        avg_q = self.total_q / self.update_count

        self._log('update.average_q', avg_q, self.step)
        self._log('update.average_loss', avg_loss, self.step)
        self._log('update.minibatch_loss', self.mb_loss, self.step)
        self._log('update.minibatch_q', self.mb_q, self.step)

    def episode(self, reward):
        self.ep_rewards.append(reward)
        self.ep_max_reward = max(self.ep_max_reward, reward)
        self.ep_min_reward = min(self.ep_min_reward, reward)

    def display(self):

        avg_loss = None if self.update_count == 0 else self.total_loss / self.update_count
        avg_q = None if self.update_count == 0 else self.total_q / self.update_count
        nonzero_reward_list = [
            reward for reward in self.ep_rewards if reward != 0]
        avg_ep_nonzero_reward = None if len(nonzero_reward_list) == 0 else sum(
            nonzero_reward_list) / float(len(nonzero_reward_list))

        values = {
            'Episode': self.ep,
            'Step': self.step,
            'Avg. Loss': avg_loss,
            'Avg. Q': avg_q,
            'Episode Avg. Reward': sum(self.ep_rewards) / float(len(self.ep_rewards)),
            'Episode Avg. Reward Non-0': avg_ep_nonzero_reward,
            'Episode Min. Reward': self.ep_min_reward,
            'Episode Max. Reward': self.ep_max_reward,
            'Minibatch Loss': self.mb_loss,
            'Minibatch Q': self.mb_q,
            'Epsilon': self.epsilon_val
        }
        print('-------')
        for key in values:
            print('{}: {}'.format(key, values[key]))

    def reset_episode(self):
        avg_ep_reward = sum(self.ep_rewards) / float(len(self.ep_rewards))
        nonzero_reward_list = [
            reward for reward in self.ep_rewards if reward != 0]
        avg_ep_nonzero_reward = sum(
            nonzero_reward_list) / float(len(nonzero_reward_list))

        self._log('ep.average_reward_nonzero', avg_ep_nonzero_reward, self.ep)
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
            self.writer.scalar_dict = {}
        elif type == 'histogram':
            self.writer.add_histogram(name, value, step)
