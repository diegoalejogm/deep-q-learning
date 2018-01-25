# deep-q-learning

PyTorch omplementation of DeepMind's **_Human-level control through deep reinforcement learning_** paper [link](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf).

<img src=".assets/net.png" width="800">

This research project proposes an general algorithm capable of learning how to play several popular **Atari** videogames, with no previous knowledge. It consists of an innovative implementation of a [Q-Learning](https://en.wikipedia.org/wiki/Q-learning) algorithm, which can find an optimal action-selection policy for any given finite Markov decision process. Such implementation uses a Deep Convolutional Neural Network to approximate the expected reward of every single states (named __Q-Values__), given a pixel-based representation of the current state of the videogame.

<img src=".assets/atari_games.png" width="800">

The name given to the usage of a Deep Neural Network in Reinforcement learning context is **Deep Reinforcement Learning**, and the name of doing Q-learning with a Deep Neural Network is **Deep Q Learning**.

The proposed implementation is found in this repository. Also, this _readme_ file contains an explanation of the paper's proposal.


## Model

### Input Representation

1. Each image frame in RGB is converted into it's [luminosity's representation] (https://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color), which reduces it to 2-dimensions.
2. Resulting images are rezised into a square image with size 84 x 84.
3. The tensor containing a sequence of the last 4 processed frames is used as input for the Deep Convolutional Neural Network. Such tensor has size (4 x 84 x 84).

### Neural Network
1. The neural network contains a sequence of convolutional layers, each followed by a [Rectifier Linear Unit (ReLU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) layer.
2. The output of the last hidden layer is flattened into a 1-Dimensional vector, which is used as the first layer of a fully-connected neural network.
3. The first layer of the fully-connected network is connected to a single hidden layer, with an additional ReLU.
4. The output layer has multiple outputs, each one for each possible action that the agent has.


## Algorithm

### __Deep Q-Learning with Experience Replay__

Pseudocode for the Q-Learning algorithm which uses the previously mentioned model is presented next.

<img src=".assets/pseudocode.png" width="700">


## Code
Upcoming Explanations...

## Experiments

For the moment, only the _Pong_ videogame has been tested, results will be uploaded soon.


## Credits
_[1] Mnih, V. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015)._
