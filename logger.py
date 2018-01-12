from tensorboardX import SummaryWriter


class Logger:

    def __init__(self):
        # TensorBoard
        self.writer = SummaryWriter()
        # Values
        self.ep_reward = 0.0
        self.eps_rewards = []
        self.grad_count = int(0)
        self.total_q = 0.0
        self.total_loss = 0.0

        # Counters
        self.update_count = 0.0

    def update(self, reward, q, loss):
        self.update_count += 1

        self.total_q += q.sum()
        self.total_loss += loss

        avg_loss = self.total_loss / self.update_count
        avg_q = self.total_q / self.update_count

        self.writer.add_scalar('avg_q', avg_q, self.update_count)
        self.writer.add_scalar('avg_loss', avg_loss, self.update_count)
        # -- LOGS
        # clear_output()
        print('Update #: {}'.format(self.update_count))
        # print('Ep. {} Reward: {}'.format(ep, ep_reward))
        # print('Eps. Rewards: {}'.format(eps_rewards[:20]))
        # print('Replay Size: {}'.format(D.count))
