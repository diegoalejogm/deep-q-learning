import gym
import tensorflow as tf
from torch import optim

from utils.learn import e_greedy_action
from utils.logger import Logger
from utils.models import ReplayMemory, History
from utils.net import DeepQNetwork, Q_targets, Q_values, save_network, copy_network, gradient_descent
from utils.processing import phi_map, tuple_to_numpy


def read_flags():
    flags = tf.app.flags
    flags.DEFINE_boolean(
        'floyd', False, 'Use directory structure for deploying in FloydHub')
    flags.DEFINE_string(
        'data_dir', './data', 'Default output data directory')
    flags.DEFINE_string(
        'log_dir', None, 'Default tensorboard data directory')
    flags.DEFINE_string(
        'in_dir', './data', 'Default input data directory')
    flags.DEFINE_integer(
        'log_freq', 1, 'Step frequency for logging')
    flags.DEFINE_boolean(
        'log_console', True, 'Step frequency for logging')
    flags.DEFINE_integer(
        'save_model_freq', 10, 'Step frequency for logging')
    FLAGS = flags.FLAGS

    # Reformat directories if using FloydHub directory structure
    if FLAGS.floyd:
        FLAGS.data_dir = '/output'
        FLAGS.log_dir = '/output'
        FLAGS.in_dir = ''
        FLAGS.log_freq = 100
        FLAGS.log_console = False
        FLAGS.save_model_freq = 100
    return FLAGS



# ----------------------------------
FLAGS = read_flags()
# ----------------------------------
# Tranining
env = gym.make('Pong-v0')
# Current iteration
step = 0
# Has trained model
has_trained_model = False
# Init training params
params = {
    'num_episodes': 2000,
    'minibatch_size': 32,
    'max_episode_length': int(10e6),  # T
    'memory_size': int(1e6)  # N
    'history_size': 4,  # k
    'train_freq': 4,
    'target_update_freq': 10000,  # C: Target nerwork update frequency
    'num_actions': env.action_space.n,
    'min_steps_train': 50000
}
# Initialize Logger
log = Logger(log_dir=FLAGS.log_dir)
# Initialize replay memory D to capacity N
D = ReplayMemory(N=params['memory_size'],
                 load_existing=True, data_dir=FLAGS.in_dir)
skip_fill_memory = D.count > 0
# Initialize action-value function Q with random weights
Q = DeepQNetwork(params['num_actions'])
log.network(Q)
# Initialize target action-value function Q^
Q_ = copy_network(Q)
# Init network optimizer
optimizer = optim.RMSprop(
    Q.parameters(), lr=0.00025, alpha=0.95, eps=.01  # ,momentum=0.95,
)
# Initialize sequence s1 = {x1} and preprocessed sequence phi = phi(s1)
H = History.initial(env)

for ep in range(params['num_episodes']):

    phi = phi_map(H.get())
    # del phi

    if (ep % FLAGS.save_model_freq) == 0:
        save_network(Q, ep, out_dir=FLAGS.data_dir)

    for _ in range(params['max_episode_length']):
        # if step % 100 == 0:
        #     print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        step += 1
        # Select action a_t for current state s_t
        action, epsilon = e_greedy_action(Q, phi, env, step)
        if step % FLAGS.log_freq == 0:
            log.epsilon(epsilon, step)
        # Execute action a_t in emulator and observe reward r_t and image x_(t+1)
        image, reward, done, _ = env.step(action)

        # Clip reward to range [-1, 1]
        reward = max(-1.0, min(reward, 1.0))
        # Set s_(t+1) = s_t, a_t, x_(t+1) and preprocess phi_(t+1) =  phi_map( s_(t+1) )
        H.add(image)
        phi_prev, phi = phi, phi_map(H.get())
        # Store transition (phi_t, a_t, r_t, phi_(t+1)) in D
        D.add((phi_prev, action, reward, phi, done))

        should_train_model = skip_fill_memory or \
            ((step > params['min_steps_train']) and
             D.can_sample(params['minibatch_size']) and
             (step % params['train_freq'] == 0))

        if should_train_model:
            if not (skip_fill_memory or has_trained_model):
                D.save(params['min_steps_train'])
            has_trained_model = True

            # Sample random minibatch of transitions ( phi_j, a_j, r_j, phi_(j+1)) from D
            phi_mb, a_mb, r_mb, phi_plus1_mb, done_mb = D.sample(
                params['minibatch_size'])
            # Perform a gradient descent step on ( y_j - Q(phi_j, a_j) )^2
            y = Q_targets(phi_plus1_mb, r_mb, done_mb, Q_)
            q_values = Q_values(Q, phi_mb, a_mb)
            q_phi, loss = gradient_descent(y, q_values, optimizer)
            # Log Loss
            if step % (params['train_freq'] * FLAGS.log_freq) == 0:
                log.q_loss(q_phi, loss, step)
            # Reset Q_
            if step % params['target_update_freq'] == 0:
                del Q_
                Q_ = copy_network(Q)

        log.episode(reward)
        # if FLAGS.log_console:
        #     log.display()

        # # Restart game if done
        if done:
            H = History.initial(env)
            log.reset_episode()
            break

writer.close()
