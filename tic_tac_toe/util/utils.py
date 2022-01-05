import json
import os
from typing import List
import pandas as pd
import numpy as np
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
        # Keys not available if MaxStrategy
        pass

    if strategy_spec['type'] == 'EpsilonGreedyStrategy':
        if start > 1:
            start = 1  # In param search Boltzmann T range might go in here
        strategy = EpsilonGreedyStrategy(start=start, end=end, decay=decay)
    elif strategy_spec['type'] == 'MaxStrategy':
        strategy = MaxStrategy()
    elif strategy_spec['type'] == 'Boltzmann':
        strategy = Boltzmann(start=start, end=end, decay=decay)
    return strategy


def memory_from_spec(memory_spec):
    try:
        capacity = memory_spec['size']
        memory = StandardReplayMemory(capacity)
        return memory
    except KeyError as e:
        print("Invalid memory specification.", e)


def _spec_to_df(spec):
    ''' Takes spec and returns a df with one row per parameter combination'''
    # Help function for flattening nested dict
    def flatten(input_dict, sep='.', prefix=''):
        out_dict = {}
        for k, v in input_dict.items():
            if isinstance(v, dict) and v:
                deeper = flatten(v, sep, prefix+k+sep)
                out_dict.update({k2: v2 for k2, v2 in deeper.items()})
            elif isinstance(v, list) and v:
                for i, l in enumerate(v, start=1):
                    if isinstance(l, dict) and l:
                        deeper = flatten(l, sep, prefix+k+sep+str(i)+sep)
                        out_dict.update({k2: v2 for k2, v2 in deeper.items()})
                    else:
                        out_dict[prefix+k] = v
            else:
                out_dict[prefix+k] = v
        return out_dict

    # Read spec to one-row df
    d = flatten(spec)
    df = pd.DataFrame.from_dict(d, orient='index').transpose()

    # Transform to one row for each possible parameter combination
    for col in df.columns:
        if not col == 'net.layers.1.config.input_shape':  # Leave input shape
            df = df.explode(col)
    df = df.reset_index(drop=True)
    return df


def df_row_to_spec(pd_series):
    ''' Takes a pandas Series (read row from spec combinations df) and
        returns a nested spec dictionary '''
    # Help function for recursively nesting flat dict by key separator
    def _nest_dict_rec(k, v, out, separator):
        k, *rest = k.split(separator, 1)
        if rest:
            _nest_dict_rec(rest[0], v, out.setdefault(k, {}), separator)
        else:
            out[k] = v

    def nest_dict(flat, separator='.'):
        result = {}
        for k, v in flat.items():
            _nest_dict_rec(k, v, result, separator)
        return result

    d = pd_series.transpose().to_dict()
    d = nest_dict(d)
    # Remove layer enumaration added in spec_to_df
    if 'layers' in d['net'].keys():
        d['net']['layers'] = [v for k, v in d['net']['layers'].items()]
    return d


def param_search_df_from_spec(spec_in):
    ''' Takes a spec file with iterable parameter values over which
        to search. Returns a pandas dataframe that holds a number of
        parameter combintations set by spec['search']['max_combinations']'''
    df = _spec_to_df(spec_in)

    if len(df) == 1:
        return df  # Always return df if no values to search over
    else:
        try:
            max_num_param_combos = spec_in['search']['max_combinations']
        except:
            raise ValueError('Set number of parameter combinations to search')

        sample_size = np.min([max_num_param_combos, len(df)])
        df_subset = df.sample(sample_size).reset_index(drop=True)
    return df_subset


def create_dir_if_not_exist(path):
    if not os.path.isdir(path):
            os.makedirs(path)
