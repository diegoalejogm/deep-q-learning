from tensorboardX import SummaryWriter


class Logger:

    def __init__(self):
        self.env_name = 'Pong-v0'
        # TensorBoard
        self.writer = SummaryWriter()
        # Episode Values
        self.ep = 0
        self.ep_rewards = []
        self.ep_max_reward = 0.0
        self.ep_min_reward = 0.0
        # Updates Values
        self.grad_count = int(0)
        self.total_q = 0.0
        self.total_loss = 0.0

        # Counters
        self.epsilon_val = 0.0
        self.update_count = 0.0
        self.step = 0.0

    def epsilon(self, eps):
        self.step += 1
        self.epsilon_val = eps
        self._log('epsilon', self.epsilon_val, self.step)

    def q_loss(self, q, loss):
        self.update_count += 1

        self.total_q += q.sum().data.cpu().numpy()[0] / int(q.size()[0])
        self.total_loss += loss.data.cpu().numpy()[0]

        avg_loss = self.total_loss / self.update_count
        avg_q = self.total_q / self.update_count

        self._log('average_q', avg_q, self.update_count)
        self._log('average_loss', avg_loss, self.update_count)

        # -- LOGS
        # clear_output()
#        print('Update #: {}'.format(self.update_count))
        # print('Ep. {} Reward: {}'.format(ep, ep_reward))
        # print('Eps. Rewards: {}'.format(eps_rewards[:20]))
        # print('Replay Size: {}'.format(D.count))

    def episode(self, reward):
        self.ep_rewards.append(reward)
        self.ep_max_reward = max(self.ep_max_reward, reward)
        self.ep_min_reward = min(self.ep_min_reward, reward)

    def display(self):
        values = {
            'Step': self.step,
            'Avg. Loss': self.total_loss / self.update_count,
            'Avg. Q': self.total_q / self.update_count,
            'Avg. Ep. Reward': sum(self.ep_rewards) / float(len(self.ep_rewards)),
            'Min. Ep. Reward': self.ep_min_reward,
            'Max. Ep. Reward': self.ep_max_reward,
            'Epsilon': self.epsilon_val
        }
        print('-------')
        for key in values:
            print('{}: {}'.format(key, values[key]))

    def reset_episode(self):
        avg_ep_reward = sum(self.ep_rewards) / float(len(self.ep_rewards))

        self._log('average_ep_reward', avg_ep_reward, self.ep)
        self._log('min_ep_reward', self.ep_min_reward, self.ep)
        self._log('max_ep_reward', self.ep_max_reward, self.ep)

        self.ep += 1
        self.ep_rewards = []
        self.ep_max_reward = 0.0
        self.ep_min_reward = 0.0

    def _log(self, name, value, step):
        self.writer.add_scalar(
            '{}/{}'.format(self.env_name, name), value, step)
