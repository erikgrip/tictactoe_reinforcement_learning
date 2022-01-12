# tictactoe_reinforcement_learning

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Training specification](#training-specification)
* [Parameter search](#parameter-search)
* [Output](#output)

## General info
In this project a deep reinforcement learning model is trained to find a winning tic tac toe strategy against an opponent playing random moves. The algorithm [Deep Q Network](https://www.tensorflow.org/agents/tutorials/0_intro_rl) (DQN) is implemented along with the exploration vs exploitating strategies 'Boltzmann' and 'Epsilon Greedy'.
The user can specify a parameter space in which to try a given number of combinations and track the results in TensorBoard.

The results an example run can be found in the output directories logs, models and specs. The training progression of that run is visible in the TensorBoard. After 5 000 training games the model scored 84% wins, 13% draws and 5% losses over 1 000 evaluation games. Not too bad!

Thanks to [sentdex](https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ) and [pythonprogramming.net](https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/) for inspiration and implementation of the ModifiedTensorBoard class.


## Technologies
Project is created with:
* Python
* Keras
* Tensorboard
* Docker


## Setup
To run this project you need Docker and Docker Compose. See official docs [here](https://docs.docker.com/compose/install/) for instructions on how to install. With Docker and docker-compose installed, clone the repository using a local terminal.
```
$ git clone https://github.com/erikgrip/tictactoe_reinforcement_learning.git
```

When the project is run it will start training and evluating models according to the [specification](#training-specification) in spec.jason in the project's top folder so make sure to read that section.

To start training and lauch a TensorBoard, navigate into the project folder and run:
```
$ docker-compose up
```
Once the TensorBoard container is running the visualisation can be accessed by navigating to http://localhost:8088 in a web browser.

The training and TensorBoard can also be launched separatly with the commands:
```
$ docker-compose up train
$ docker-compose up tensorboard
```


## Training Specification
The file spec.json in the project's top folder specifies the parameters of the training. It should contain the following entries.
### Run
Example:
```
"run": {
        "num_train_episodes": 5000,
        "num_eval_episodes": 1000
    }
```
**num_train_episodes**: The number of games played to train the model.
**num_eval_episodes**: The number of games played with learning and exploration turned off. These games are played after the training games are completed.

### Search
Example:
```
"search": {
        "max_combinations": 1
    }
```
**max_combinations**: The number of combinations to use in a [parameter search](#parameter-search) in training. If the combinations in the specifications are fewer than this number, then all possible combinations will be used.

### Strategy
Example:
```
"strategy": {
    "type": "Boltzmann",
    "max": 1,
    "min": 0.05,
    "decay": 0.005
    }
```
**type**: The exploration approach to use. One of Boltzmann, EpsilonGreedyStrategy or MaxStrategy.
**max**: Exploration value at start of training.
**min**: The amount of exploration not to go under in training.
**decay**: Sets how fast the exploration rate will drop drop max to min, in termes of games played.

To get a feel for theses parameters, try a few different values out and watch the outcome 'exploration_parameter' in TensorBoard.

### Net
The architecture and parameters of the neural networks.
The following example would create a network with an input layer, one hidden Dense layer and an output layer.
```
"net": {
        "name": "2xDen",
        "layers": [{
            "class_name": "Flatten",
            "config": {
                "input_shape": [3, 3, 1]}
            },
            {
            "class_name": "Dense",
            "config": {
                "units": 16,
                "activation": "relu",
                "kernel_initializer": "glorot_uniform"}
            },
            {
            "class_name": "Dense",
            "config": {
                "units": 9,
                "activation": "linear"}
            }],
        "lr": 0.001,
        "loss": "mean_squared_error",
        "keras_version": "2.3.1"
```
**name**: Used to name saved files to make them easier to tell apart.
**layers**: The buildning blocks of the neural network. See [Keras docs](https://keras.io/api/layers/#core-layers) for layer types and their config options.
**lr**: The learning rate.
**loss**: The loss function.
**keras_version**: Should be 2.3.1 unless updated in requirements.txt

### Replay Memory
The replay memory holds the samples used in training.
Example:
```
"replay_memory": {
        "size": 10000,
        "minibatch_size": 128,
        "min_memory": 256
    }
```
**size**: Maximum number of samples the memory will hold before removing old observations.
**minibatch_size**: Number of samples used at once in training.
**min_memory**: Number of observations to collect before training starts. Must be at least as large as the minibatch_size.

### Algorithm
Example:
```
"algorithm": {
        "target_net_update_freq": 100,
        "discount": 0.99
    }
```
**target_net_update_freq**: Number of episodes to play before updating the target net's weights.
**discount**: The rate of which expected future rewards diminish compared to instant rewards. In this game there's only one reward (or penalty) at the end of each game though.

## Parameter search
To train multiple models using different parameters, set the parameter choises in square brackets in the spec, like so:
```
"strategy": {
    "type": ["Boltzmann", "EpsilonGreedyStrategy"],
    "max": 1,
    "min": [0.05, 0.1, 0.2],
    "decay": [0.001, 0.005]
    }
```

## Output
Each training run will store logs, models and specifications in their respective directory. The logs are read by Tensorboard, and the saved specs documents the parameter settings of the individual models.
