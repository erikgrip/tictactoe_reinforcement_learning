import json
from keras.models import  Sequential, model_from_json
from keras.optimizers import Adam
from keras.layers import Activation, Conv2D, Dense, Flatten, Dropout
from tic_tac_toe.agent.strategy.epsilon_greedy import EpsilonGreedyStrategy
from tic_tac_toe.agent.strategy.boltzmann import Boltzmann
from tic_tac_toe.agent.strategy.max_strategy import MaxStrategy
from tic_tac_toe.agent.memory.replay import StandardReplayMemory
from keras.models import load_model

def sequential_model_from_spec(net_spec):

    if 'load_path' in net_spec:
        try:
            model = load_model(net_spec['load_path'])
        except:
            raise ValueError(f"Couldn't load model {net_spec['load_path']}")
    else:
        try:
            d = {}
            d['class_name'] = 'Sequential'
            d['config'] = {'name': net_spec['name'],
                           'layers': net_spec['layers']}
            d['keras_version'] = net_spec['keras_version']
            d['backend'] = 'tensorflow'

            j = json.dumps(d)
            model = model_from_json(j)
            model.compile(optimizer=Adam(lr=net_spec['lr']),
                          loss=net_spec['loss'],
                          metrics=['accuracy'])  # Make dynamic?
        except:
            raise ValueError('Invalid net specification')
    return model


def strategy_from_spec(strategy_spec):
    try:
        start = strategy_spec['max']
        end = strategy_spec['min']
        decay = strategy_spec['decay']
    except:
        pass
        
    if strategy_spec['type'] == 'EpsilonGreedyStrategy':
        strategy = EpsilonGreedyStrategy(start=start, end=end, decay=decay)
    elif strategy_spec['type'] == 'MaxStrategy':
        strategy = MaxStrategy()
    elif strategy_spec['type'] == 'Boltzmann':
        strategy = Boltzmann(start=start, end=end, decay=decay)
    return strategy


def memory_from_spec(memory_spec):
    try:
        capacity = memory_spec['size']
    except:
        pass  # Fix later

    if memory_spec['type'] == 'StandardReplayMemory':
        memory = StandardReplayMemory(capacity)
    else:
        raise ValueError("Invalid memory specification")
    return memory