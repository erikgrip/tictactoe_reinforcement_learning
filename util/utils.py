import json
from keras.models import  Sequential, model_from_json
from keras.optimizers import Adam
from keras.layers import Activation, Conv2D, Dense, Flatten, Dropout
from tic_tac_toe.Strategy import EpsilonGreedyStrategy, Boltzmann, MaxStrategy

def sequential_model_from_spec(net_spec):
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

def strategy_model_from_spec(strategy_spec):
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