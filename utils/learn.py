import numpy as np


def e_greedy_action(Q, phi, env, step):
    # Initial values
    initial_epsilon, final_epsilon = 1.0, .1
    # Decay steps
    decay_steps = float(1e6)
    # Calculate step size to move from final to initial epsilon with #decay_steps
    step_size = (initial_epsilon - final_epsilon) / decay_steps
    # Calculate annealed epsilon
    ann_eps = initial_epsilon - step * step_size
    # Define allowsd min. epsilon
    min_eps = 0.1
    # Set epsilon as max(min_eps, annealed_epsilon)
    epsilon = max(min_eps, ann_eps)

    # Obtain a random value in range [0,1)
    rand = np.random.uniform()

    print(Q(phi))

    # With probability e select random action a_t
    if rand < epsilon:
        return env.action_space.sample(), epsilon

    else:
        # Otherwise select action that maximises Q(phi)
        # In other words: a_t = argmax_a Q(phi, a)
        return Q(phi).max(1)[1].data, epsilon
